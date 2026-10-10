//! REMOTE MEDIA URLS (issue #9): fetch an `http(s)://` image, audio or video reference named in a
//! request. OFF by default — `arf serve` has no authentication, so a fetch is made on behalf of
//! whoever can reach the port. `ARF_MEDIA_URLS=1` turns it on, and every fetch is bounded:
//!
//! - only `http` and `https`;
//! - every address the host resolves to must be PUBLIC (no loopback, private, link-local, CGNAT,
//!   unique-local, multicast, documentation or reserved range, and the IPv4 inside a mapped or
//!   NAT64 IPv6 address is checked too). The vetted addresses are the ones connected to, so a
//!   second DNS answer cannot swap in a private one (DNS rebinding);
//! - redirects are NOT followed (a redirect is how a public URL reaches a private address);
//! - [`MAX_BYTES`] per fetch, checked against `Content-Length` and again while reading;
//! - [`TIMEOUT`] for the whole request.
//!
//! No proxy from the environment is used. The fetch blocks its thread, as the ffmpeg decode of a
//! video already does.

use std::io::Read;
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr, ToSocketAddrs};
use std::time::Duration;

/// Largest body fetched, per reference.
pub const MAX_BYTES: u64 = 64 << 20;

/// Whole-request timeout (connect, headers and body).
pub const TIMEOUT: Duration = Duration::from_secs(20);

/// `http://` or `https://` (case-insensitive scheme).
pub fn is_remote(url: &str) -> bool {
    let l = url.get(..8).unwrap_or(url).to_ascii_lowercase();
    l.starts_with("http://") || l.starts_with("https://")
}

/// `ARF_MEDIA_URLS` is set.
fn enabled() -> bool {
    std::env::var_os("ARF_MEDIA_URLS").is_some_and(|v| !v.is_empty() && v != "0")
}

/// Fetch a remote `what` ("image", "audio", "video") reference, or say why not.
pub fn fetch(url: &str, what: &str) -> Result<Vec<u8>, String> {
    if !enabled() {
        return Err(format!(
            "remote {what} URLs are not fetched (set ARF_MEDIA_URLS=1 on the server to allow; \
             or send a data: URL)"
        ));
    }
    let agent = ureq::AgentBuilder::new()
        .redirects(0)
        .timeout(TIMEOUT)
        .resolver(public_only)
        .build();
    let resp = agent
        .get(url)
        .call()
        .map_err(|e| format!("{what} fetch {url}: {e}"))?;
    let status = resp.status();
    if !(200..300).contains(&status) {
        return Err(if (300..400).contains(&status) {
            format!("{what} fetch {url}: HTTP {status} (redirects are not followed)")
        } else {
            format!("{what} fetch {url}: HTTP {status}")
        });
    }
    if let Some(n) = resp
        .header("content-length")
        .and_then(|v| v.parse::<u64>().ok())
    {
        if n > MAX_BYTES {
            return Err(too_big(what, url));
        }
    }
    let mut body = Vec::new();
    resp.into_reader()
        .take(MAX_BYTES + 1)
        .read_to_end(&mut body)
        .map_err(|e| format!("{what} fetch {url}: {e}"))?;
    if body.len() as u64 > MAX_BYTES {
        return Err(too_big(what, url));
    }
    Ok(body)
}

fn too_big(what: &str, url: &str) -> String {
    format!("{what} fetch {url}: larger than {} MB", MAX_BYTES >> 20)
}

/// The fetch's resolver: every address `netloc` resolves to, or an error if ANY is not public.
fn public_only(netloc: &str) -> std::io::Result<Vec<SocketAddr>> {
    let addrs: Vec<SocketAddr> = netloc.to_socket_addrs()?.collect();
    if let Some(a) = addrs.iter().find(|a| !is_public(a.ip())) {
        return Err(std::io::Error::new(
            std::io::ErrorKind::PermissionDenied,
            format!("{netloc} resolves to a non-public address ({})", a.ip()),
        ));
    }
    Ok(addrs)
}

/// A globally routable unicast address (what `IpAddr::is_global` will say once stable).
pub fn is_public(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => is_public_v4(v4),
        IpAddr::V6(v6) => is_public_v6(v6),
    }
}

fn is_public_v4(ip: Ipv4Addr) -> bool {
    let [a, b, c, _] = ip.octets();
    !(ip.is_unspecified()
        || ip.is_loopback()
        || ip.is_private()
        || ip.is_link_local()
        || ip.is_broadcast()
        || ip.is_multicast()
        || ip.is_documentation()
        || a == 0 // "this network"
        || (a == 100 && (64..128).contains(&b)) // CGNAT 100.64/10
        || (a == 192 && b == 0 && c == 0) // IETF protocol assignments
        || (a == 198 && (b == 18 || b == 19)) // benchmarking 198.18/15
        || a >= 240) // reserved
}

fn is_public_v6(ip: Ipv6Addr) -> bool {
    if let Some(v4) = ip.to_ipv4_mapped() {
        return is_public_v4(v4);
    }
    let s = ip.segments();
    if s[..6] == [0x64, 0xff9b, 0, 0, 0, 0] {
        // NAT64: the IPv4 it reaches.
        let [a, b] = s[6].to_be_bytes();
        let [c, d] = s[7].to_be_bytes();
        return is_public_v4(Ipv4Addr::new(a, b, c, d));
    }
    !(ip.is_unspecified()
        || ip.is_loopback()
        || ip.is_multicast()
        || (s[0] & 0xfe00) == 0xfc00 // unique local fc00::/7
        || (s[0] & 0xffc0) == 0xfe80 // link-local fe80::/10
        || (s[0] & 0xffc0) == 0xfec0 // site-local fec0::/10 (deprecated)
        || (s[0] == 0x2001 && s[1] == 0x0db8) // documentation
        || s[..6] == [0, 0, 0, 0, 0, 0]) // IPv4-compatible (deprecated)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Addresses are built from octets (the repository's leak scan refuses dotted-quad literals).
    fn v4(a: u8, b: u8, c: u8, d: u8) -> IpAddr {
        IpAddr::V4(Ipv4Addr::new(a, b, c, d))
    }

    #[test]
    fn private_and_reserved_addresses_are_refused() {
        let v4s = [
            v4(127, 0, 0, 1),
            v4(10, 1, 2, 3),
            v4(172, 16, 0, 1),
            v4(192, 168, 1, 1),
            v4(169, 254, 169, 254), // cloud metadata
            v4(100, 64, 0, 1),
            v4(0, 0, 0, 0),
            v4(255, 255, 255, 255),
            v4(224, 0, 0, 1),
            v4(198, 18, 0, 1),
            v4(192, 0, 0, 8),
            v4(240, 0, 0, 1),
        ];
        let v6s = [
            "::1",
            "::",
            "fc00::1",
            "fd12:3456::1",
            "fe80::1",
            "ff02::1",
            "2001:db8::1",
            "::ffff:7f00:1",      // mapped loopback
            "::ffff:a00:1",       // mapped 10/8
            "64:ff9b::a9fe:a9fe", // NAT64 of the metadata address
            "::7f00:1",           // IPv4-compatible loopback
        ];
        for a in v4s
            .into_iter()
            .chain(v6s.iter().map(|a| a.parse().unwrap()))
        {
            assert!(!is_public(a), "{a} must be refused");
        }
        for a in [
            v4(1, 1, 1, 1),
            v4(8, 8, 8, 8),
            v4(93, 184, 215, 14),
            "2606:4700:4700::1111".parse().unwrap(),
            "::ffff:808:808".parse().unwrap(), // mapped, public
        ] {
            assert!(is_public(a), "{a} is public");
        }
    }

    #[test]
    fn the_resolver_refuses_a_name_with_any_private_answer() {
        let e = public_only(&SocketAddr::new(v4(127, 0, 0, 1), 80).to_string()).unwrap_err();
        assert_eq!(e.kind(), std::io::ErrorKind::PermissionDenied);
        assert!(public_only("[::1]:443").is_err());
        let public = SocketAddr::new(v4(1, 1, 1, 1), 443);
        assert_eq!(public_only(&public.to_string()).unwrap(), vec![public]);
    }

    #[test]
    fn only_http_and_https_are_remote() {
        assert!(is_remote("https://example.com/a.png"));
        assert!(is_remote("HTTP://example.com/a.png"));
        for u in [
            "data:image/png;base64,AA==",
            "file:///a.png",
            "/a.png",
            "ftp://x/a",
            "http",
        ] {
            assert!(!is_remote(u), "{u}");
        }
    }

    /// Over the network, so ignored by default: `ARF_MEDIA_URLS=1 cargo test -p arf-serve --lib
    /// remote_media -- --ignored`. A public PNG arrives whole; a redirect (http -> https) is
    /// refused rather than followed.
    #[test]
    #[ignore]
    fn a_public_png_is_fetched_and_a_redirect_is_not_followed() {
        assert!(enabled(), "set ARF_MEDIA_URLS=1");
        let png = fetch(
            "https://github.githubassets.com/favicons/favicon.png",
            "image",
        )
        .unwrap();
        assert!(
            png.starts_with(b"\x89PNG"),
            "{} bytes, not a PNG",
            png.len()
        );
        let e = fetch("http://github.com/", "image").unwrap_err();
        assert!(e.contains("redirects are not followed"), "{e}");
        let e = fetch("http://localhost:9/x.png", "image").unwrap_err();
        assert!(e.contains("non-public"), "{e}");
    }

    /// Off by default: the refusal names the switch. (A test binary that sets ARF_MEDIA_URLS
    /// skips this.)
    #[test]
    fn fetching_is_off_by_default() {
        if enabled() {
            eprintln!("NOTE: ARF_MEDIA_URLS is set; skipping the default-off check");
            return;
        }
        let e = fetch("https://example.com/a.png", "image").unwrap_err();
        assert!(e.contains("ARF_MEDIA_URLS=1"), "{e}");
    }
}
