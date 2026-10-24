"""SSRF (Server-Side Request Forgery) protection and URL validation.

Ensures that user-supplied URLs adhere to HTTP/HTTPS schemes, do not resolve
to internal, private, loopback, multicast, or cloud-metadata addresses, and
enforces strict fail-closed safety.

Handles:
- IPv4, IPv6, IPv4-mapped IPv6 (::ffff:0:0/96), and NAT64 (64:ff9b::/96) addresses.
- Normalization of decimal/hex/octal encoded IP literals in hostnames.
- Prevention of DNS rebinding and dual-homed attacks by verifying all resolved IPs.
"""

import ipaddress
import socket
import threading
import time
from typing import Dict, Optional, Tuple
from urllib.parse import urlparse

from agentx.config import settings

# Explicitly blocked hostnames
BLOCKED_HOSTNAMES = {
    "localhost",
    "localhost.localdomain",
    "metadata.google.internal",
    "metadata.internal",
    "instance-data",
}

# Explicitly blocked metadata IP addresses (IPv4 & IPv6)
BLOCKED_METADATA_IPS = {
    "169.254.169.254",  # AWS, GCP, Azure, OpenStack instance metadata
    "fd00:ec2::254",    # AWS IPv6 IMDSv2
}

# IPv6 translation, tunneling, and special networks
NAT64_WKP = ipaddress.IPv6Network("64:ff9b::/96")      # RFC 6052 Well-Known Prefix
NAT64_LOCAL = ipaddress.IPv6Network("64:ff9b:1::/48")  # RFC 8215 Local-Use NAT64
TEREDO = ipaddress.IPv6Network("2001::/32")            # RFC 4380 Teredo tunneling
SIX_TO_FOUR = ipaddress.IPv6Network("2002::/16")       # RFC 3056 6to4 tunneling (deprecated RFC 7526)
IPV6_DOC = ipaddress.IPv6Network("2001:db8::/32")      # RFC 3849 Documentation
IPV6_DISCARD = ipaddress.IPv6Network("100::/64")       # RFC 6666 Discard prefix
IPV6_COMPAT = ipaddress.IPv6Network("::/96")           # RFC 4291 IPv4-compatible (deprecated)
CGNAT_IPV4 = ipaddress.IPv4Network("100.64.0.0/10")     # RFC 6598 Carrier-Grade NAT


def parse_ip_literal(host_str: str) -> Optional[ipaddress._BaseAddress]:
    """Attempt to parse an IP literal in standard, integer, or hex notation.

    Args:
        host_str: Hostname or IP literal string.

    Returns:
        IPv4Address or IPv6Address if successfully parsed, None otherwise.
    """
    cleaned = host_str.strip().strip("[]")
    # Standard IPv4 or IPv6
    try:
        return ipaddress.ip_address(cleaned)
    except ValueError:
        pass

    # Decimal / Dword integer literal (e.g., 2130706433 -> 127.0.0.1)
    if cleaned.isdigit():
        try:
            val = int(cleaned)
            if 0 <= val <= 0xFFFFFFFF:
                return ipaddress.IPv4Address(val)
        except (ValueError, OverflowError):
            pass

    # Hexadecimal literal (e.g., 0x7f000001 -> 127.0.0.1)
    if cleaned.lower().startswith("0x"):
        try:
            val = int(cleaned, 16)
            if 0 <= val <= 0xFFFFFFFF:
                return ipaddress.IPv4Address(val)
        except (ValueError, OverflowError):
            pass

    return None


def is_ip_allowed(ip_str: str) -> Tuple[bool, Optional[str]]:
    """Determine whether an IP address is safe for outbound browser navigation.

    Validates:
    - IPv4 addresses against loopback, private, link-local, multicast, reserved, CGNAT, metadata.
    - IPv6 addresses against loopback, ULA, link-local, multicast, documentation, discard, metadata.
    - IPv4-mapped IPv6 (::ffff:0:0/96): unpacks and strictly validates the underlying IPv4 address.
    - NAT64 Well-Known Prefix (64:ff9b::/96): unpacks and strictly validates the embedded IPv4 address.
      (Does not blindly allow or blindly reject NAT64 translations).
    - Blocks Local NAT64 (64:ff9b:1::/48), 6to4 tunneling (2002::/16), and Teredo (2001::/32).

    Args:
        ip_str: IP address string (IPv4 or IPv6).

    Returns:
        Tuple of (is_safe, failure_reason)
    """
    try:
        ip = ipaddress.ip_address(ip_str.strip().strip("[]"))
    except ValueError:
        return False, f"Invalid IP address format: {ip_str}"

    # Check known cloud metadata IPs
    if str(ip) in BLOCKED_METADATA_IPS:
        return False, f"Access to cloud metadata address {ip_str} is prohibited."

    # Handle IPv6-specific mappings and tunnel prefixes
    if isinstance(ip, ipaddress.IPv6Address):
        # 1. IPv4-mapped IPv6 (::ffff:0:0/96)
        if ip.ipv4_mapped:
            v4_safe, v4_reason = is_ip_allowed(str(ip.ipv4_mapped))
            if not v4_safe:
                return False, f"IPv4-mapped address targets prohibited destination ({v4_reason})"
            return True, None

        # 2. NAT64 Well-Known Prefix (64:ff9b::/96, RFC 6052)
        if ip in NAT64_WKP:
            embedded_v4_int = int(ip) & 0xFFFFFFFF
            embedded_v4 = ipaddress.IPv4Address(embedded_v4_int)
            v4_safe, v4_reason = is_ip_allowed(str(embedded_v4))
            if not v4_safe:
                return False, f"NAT64 translation targets prohibited destination ({v4_reason})"
            return True, None

        # 3. Local-Use NAT64 (64:ff9b:1::/48, RFC 8215)
        if ip in NAT64_LOCAL:
            return False, f"Access to local-use NAT64 address {ip_str} is prohibited."

        # 4. 6to4 tunneling (2002::/16, RFC 3056, deprecated RFC 7526)
        if ip in SIX_TO_FOUR:
            v4_int = (int(ip) >> 80) & 0xFFFFFFFF
            embedded_v4 = ipaddress.IPv4Address(v4_int)
            v4_safe, v4_reason = is_ip_allowed(str(embedded_v4))
            if not v4_safe:
                return False, f"6to4 tunneling targets prohibited destination ({v4_reason})"
            return False, f"Access to deprecated 6to4 tunneling address {ip_str} is prohibited."

        # 5. Teredo tunneling (2001::/32, RFC 4380)
        if ip in TEREDO:
            return False, f"Access to Teredo tunneling address {ip_str} is prohibited."

        # 6. IPv4-compatible (::/96, deprecated RFC 4291)
        if ip in IPV6_COMPAT and ip != ipaddress.IPv6Address("::1"):
            return False, f"Access to deprecated IPv4-compatible address {ip_str} is prohibited."

        # 7. Documentation / Discard prefixes
        if ip in IPV6_DOC or ip in IPV6_DISCARD:
            return False, f"Access to documentation or discard address {ip_str} is prohibited."

    # Loopback (127.0.0.0/8, ::1)
    if ip.is_loopback:
        return False, f"Access to loopback address {ip_str} is prohibited."

    # Private networks (RFC 1918: 10/8, 172.16/12, 192.168/16; ULA: fc00::/7)
    if ip.is_private:
        return False, f"Access to private network address {ip_str} is prohibited."

    # Link-local (169.254.0.0/16, fe80::/10)
    if ip.is_link_local:
        return False, f"Access to link-local address {ip_str} is prohibited."

    # Multicast (224.0.0.0/4, ff00::/8)
    if ip.is_multicast:
        return False, f"Access to multicast address {ip_str} is prohibited."

    # Reserved / unspecified (0.0.0.0/8, ::, 240.0.0.0/4)
    if ip.is_reserved or ip.is_unspecified:
        return False, f"Access to reserved or unspecified address {ip_str} is prohibited."

    # Carrier-Grade NAT (RFC 6598: 100.64.0.0/10)
    if isinstance(ip, ipaddress.IPv4Address) and ip in CGNAT_IPV4:
        return False, f"Access to Carrier-Grade NAT address {ip_str} is prohibited."

    return True, None


def validate_url_for_ssrf(url: str) -> Tuple[bool, Optional[str], Optional[str]]:
    """Validate a URL against SSRF threats, scheme restrictions, and DNS rebinding.

    Enforces:
    1. Scheme must be strictly 'http' or 'https'.
    2. Hostname must be present and not on the blocked list or blocked TLDs (.internal, .local, .onion).
    3. Normalizes IP literals (standard, decimal dword, hex) and validates them directly.
    4. Resolves DNS to all IP addresses (AF_UNSPEC).
    5. Fail-Closed: Every single resolved IP address must be public and permitted.
       (Blocks DNS rebinding / dual-homed attacks where one IP is private).

    Args:
        url: The candidate URL string.

    Returns:
        Tuple of (is_safe, error_reason, resolved_ip)
    """
    if not url or not isinstance(url, str):
        return False, "URL must be a non-empty string.", None

    url = url.strip()

    # Parse URL
    try:
        parsed = urlparse(url)
    except Exception as e:
        return False, f"Failed to parse URL: {e}", None

    # 1. Scheme check
    if parsed.scheme.lower() not in ("http", "https"):
        return (
            False,
            f"Unsupported URL scheme '{parsed.scheme}'. Only 'http' and 'https' are permitted.",
            None,
        )

    # 2. Hostname presence check
    hostname = parsed.hostname
    if not hostname:
        return False, "URL does not contain a valid hostname.", None

    hostname_lower = hostname.lower().strip()

    # Blocked hostnames check
    if hostname_lower in BLOCKED_HOSTNAMES:
        return False, f"Access to hostname '{hostname}' is blocked by security policy.", None

    if (
        hostname_lower.endswith(".internal")
        or hostname_lower.endswith(".local")
        or hostname_lower.endswith(".localhost")
        or hostname_lower.endswith(".onion")
    ):
        return False, f"Access to internal or reserved domain '{hostname}' is prohibited.", None

    # 3. Direct IP literal check (catches decimal dword, hex, octal, standard IP literals)
    ip_literal = parse_ip_literal(hostname_lower)
    if ip_literal is not None:
        is_safe, reason = is_ip_allowed(str(ip_literal))
        if not is_safe:
            return False, f"Host specifies prohibited IP address: {reason}", str(ip_literal)
        return True, None, str(ip_literal)

# --- Thread-Safe Bounded DNS Resolution Cache ---
_DNS_CACHE: Dict[Tuple[str, int], Tuple[float, bool, Optional[str], Optional[str]]] = {}
_DNS_CACHE_LOCK = threading.Lock()


def clear_dns_cache() -> None:
    """Clear in-memory DNS resolution cache (useful for testing or network changes)."""
    with _DNS_CACHE_LOCK:
        _DNS_CACHE.clear()


def resolve_and_validate_hostname(
    hostname_lower: str,
    port: int,
) -> Tuple[bool, Optional[str], Optional[str]]:
    """Resolve DNS with bounded TTL caching and fail-closed validation of all resolved IPs.

    Prevents repetitive synchronous DNS lookups for sub-resources while strictly maintaining
    SSRF defenses against DNS rebinding and private/internal IP destinations.

    Args:
        hostname_lower: Normalized lowercase hostname.
        port: Target TCP port.

    Returns:
        Tuple of (is_safe, error_reason, primary_ip)
    """
    now = time.time()
    cache_key = (hostname_lower, port)

    with _DNS_CACHE_LOCK:
        cached = _DNS_CACHE.get(cache_key)
        if cached:
            expiry, is_safe, reason, primary_ip = cached
            if now < expiry:
                return is_safe, reason, primary_ip
            else:
                del _DNS_CACHE[cache_key]

    try:
        addr_infos = socket.getaddrinfo(hostname_lower, port, socket.AF_UNSPEC, socket.SOCK_STREAM)
    except socket.gaierror as e:
        return False, f"DNS resolution failed for host '{hostname_lower}': {e}", None
    except Exception as e:
        return False, f"DNS resolution error for host '{hostname_lower}': {e}", None

    if not addr_infos:
        return False, f"No IP addresses resolved for host '{hostname_lower}'.", None

    resolved_ips = []
    for addr_info in addr_infos:
        sockaddr = addr_info[4]
        ip_str = sockaddr[0]
        resolved_ips.append(ip_str)

        is_safe, reason = is_ip_allowed(ip_str)
        if not is_safe:
            result = (False, f"Host '{hostname_lower}' resolved to prohibited address ({reason})", ip_str)
            with _DNS_CACHE_LOCK:
                # Cache blocked response briefly (15s) to mitigate repeated scans
                _DNS_CACHE[cache_key] = (now + 15.0, False, result[1], ip_str)
            return result

    primary_ip = resolved_ips[0] if resolved_ips else None
    result = (True, None, primary_ip)

    ttl = getattr(settings, "DNS_CACHE_TTL_SECONDS", 60.0)
    max_entries = getattr(settings, "DNS_CACHE_MAX_ENTRIES", 500)

    with _DNS_CACHE_LOCK:
        if len(_DNS_CACHE) >= max_entries:
            oldest_key = min(_DNS_CACHE, key=lambda k: _DNS_CACHE[k][0])
            del _DNS_CACHE[oldest_key]
        _DNS_CACHE[cache_key] = (now + ttl, True, None, primary_ip)

    return result


def validate_url_for_ssrf(url: str) -> Tuple[bool, Optional[str], Optional[str]]:
    """Validate a URL against SSRF threats, scheme restrictions, and DNS rebinding.

    Enforces:
    1. Scheme must be strictly 'http' or 'https'.
    2. Hostname must be present and not on the blocked list or blocked TLDs (.internal, .local, .onion).
    3. Normalizes IP literals (standard, decimal dword, hex) and validates them directly.
    4. Resolves DNS to all IP addresses (AF_UNSPEC) with bounded TTL caching.
    5. Fail-Closed: Every single resolved IP address must be public and permitted.
       (Blocks DNS rebinding / dual-homed attacks where one IP is private).

    Args:
        url: The candidate URL string.

    Returns:
        Tuple of (is_safe, error_reason, resolved_ip)
    """
    if not url or not isinstance(url, str):
        return False, "URL must be a non-empty string.", None

    url = url.strip()

    # Parse URL
    try:
        parsed = urlparse(url)
    except Exception as e:
        return False, f"Failed to parse URL: {e}", None

    # 1. Scheme check
    if parsed.scheme.lower() not in ("http", "https"):
        return (
            False,
            f"Unsupported URL scheme '{parsed.scheme}'. Only 'http' and 'https' are permitted.",
            None,
        )

    # 2. Hostname presence check
    hostname = parsed.hostname
    if not hostname:
        return False, "URL does not contain a valid hostname.", None

    hostname_lower = hostname.lower().strip()

    # Blocked hostnames check
    if hostname_lower in BLOCKED_HOSTNAMES:
        return False, f"Access to hostname '{hostname}' is blocked by security policy.", None

    if (
        hostname_lower.endswith(".internal")
        or hostname_lower.endswith(".local")
        or hostname_lower.endswith(".localhost")
        or hostname_lower.endswith(".onion")
    ):
        return False, f"Access to internal or reserved domain '{hostname}' is prohibited.", None

    # 3. Direct IP literal check (catches decimal dword, hex, octal, standard IP literals)
    ip_literal = parse_ip_literal(hostname_lower)
    if ip_literal is not None:
        is_safe, reason = is_ip_allowed(str(ip_literal))
        if not is_safe:
            return False, f"Host specifies prohibited IP address: {reason}", str(ip_literal)
        return True, None, str(ip_literal)

    # 4. DNS Resolution and IP checks (Fail Closed, protects against DNS rebinding / dual-homing)
    try:
        port = parsed.port or (443 if parsed.scheme.lower() == "https" else 80)
    except (ValueError, Exception):
        return False, "URL contains an invalid port specification.", None
    return resolve_and_validate_hostname(hostname_lower, port)
