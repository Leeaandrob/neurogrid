// Package p2p provides libp2p-based peer-to-peer networking for NeuroGrid.
package p2p

import (
	"context"
	"fmt"
	"log"

	"github.com/libp2p/go-libp2p"
	"github.com/libp2p/go-libp2p/core/host"
	"github.com/libp2p/go-libp2p/core/network"
	"github.com/libp2p/go-libp2p/core/peer"
	"github.com/libp2p/go-libp2p/p2p/protocol/circuitv2/relay"
	"github.com/multiformats/go-multiaddr"
)

// HostConfig holds configuration for creating a libp2p host.
type HostConfig struct {
	ListenPort     int
	BootstrapPeers []peer.AddrInfo
	EnableRelay    bool
	StaticRelays   []peer.AddrInfo
}

// DefaultHostConfig returns a default host configuration.
func DefaultHostConfig(port int) HostConfig {
	return HostConfig{
		ListenPort:   port,
		EnableRelay:  true,
		StaticRelays: nil, // Will use public relays if nil
	}
}

// NewHost creates a new libp2p host with TCP and QUIC transports.
func NewHost(ctx context.Context, listenPort int) (host.Host, error) {
	return NewHostWithConfig(ctx, DefaultHostConfig(listenPort))
}

// NewHostWithConfig creates a new libp2p host with the given configuration.
// Includes NAT traversal features: AutoNAT, hole punching, relay, and UPnP.
func NewHostWithConfig(ctx context.Context, cfg HostConfig) (host.Host, error) {
	// Build listen addresses
	tcpAddr := fmt.Sprintf("/ip4/0.0.0.0/tcp/%d", cfg.ListenPort)
	quicAddr := fmt.Sprintf("/ip4/0.0.0.0/udp/%d/quic-v1", cfg.ListenPort)

	opts := []libp2p.Option{
		libp2p.ListenAddrStrings(tcpAddr, quicAddr),

		// NAT traversal options
		libp2p.NATPortMap(),          // UPnP/NAT-PMP port mapping
		libp2p.EnableNATService(),    // AutoNAT service for NAT detection
		libp2p.EnableHolePunching(),  // Direct connection through NAT
		libp2p.EnableRelay(),         // Circuit relay for fallback
	}

	// Configure AutoRelay with static relays if provided
	if cfg.EnableRelay && len(cfg.StaticRelays) > 0 {
		opts = append(opts,
			libp2p.EnableAutoRelayWithStaticRelays(cfg.StaticRelays),
		)
	} else if cfg.EnableRelay {
		// Use default relay discovery
		opts = append(opts,
			libp2p.EnableAutoRelay(),
		)
	}

	h, err := libp2p.New(opts...)
	if err != nil {
		return nil, fmt.Errorf("failed to create libp2p host: %w", err)
	}

	// Log NAT status
	go logNATStatus(ctx, h)

	return h, nil
}

// EnableRelayService enables this host to act as a relay for other peers.
// Use this on bootstrap/relay nodes that have public IP addresses.
func EnableRelayService(h host.Host) error {
	_, err := relay.New(h)
	if err != nil {
		return fmt.Errorf("failed to enable relay service: %w", err)
	}
	log.Printf("Relay service enabled on host %s", h.ID())
	return nil
}

// logNATStatus logs the NAT reachability status changes.
func logNATStatus(ctx context.Context, h host.Host) {
	sub, err := h.EventBus().Subscribe(new(network.Reachability))
	if err != nil {
		log.Printf("Failed to subscribe to reachability events: %v", err)
		return
	}
	defer sub.Close()

	for {
		select {
		case <-ctx.Done():
			return
		case ev := <-sub.Out():
			if reachability, ok := ev.(network.Reachability); ok {
				log.Printf("NAT reachability changed: %s", reachability)
			}
		}
	}
}

// GetNATStatus returns the current NAT reachability status.
func GetNATStatus(h host.Host) network.Reachability {
	// Get observed addresses count as indicator
	addrs := h.Addrs()
	if len(addrs) == 0 {
		return network.ReachabilityUnknown
	}

	// Check if we have public addresses
	for _, addr := range addrs {
		if isPublicAddr(addr) {
			return network.ReachabilityPublic
		}
	}

	return network.ReachabilityPrivate
}

// isPublicAddr checks if an address appears to be publicly routable.
func isPublicAddr(addr multiaddr.Multiaddr) bool {
	// Check for non-private IP addresses
	protocols := addr.Protocols()
	for _, p := range protocols {
		if p.Code == multiaddr.P_IP4 || p.Code == multiaddr.P_IP6 {
			ip, err := addr.ValueForProtocol(p.Code)
			if err == nil && ip != "" {
				// Simple check for non-private ranges
				if ip != "127.0.0.1" && ip != "::1" &&
					!hasPrefix(ip, "192.168.") &&
					!hasPrefix(ip, "10.") &&
					!hasPrefix(ip, "172.") {
					return true
				}
			}
		}
	}
	return false
}

func hasPrefix(s, prefix string) bool {
	return len(s) >= len(prefix) && s[:len(prefix)] == prefix
}

// GetFullAddrs returns the host's addresses with peer ID appended.
func GetFullAddrs(h host.Host) []multiaddr.Multiaddr {
	peerID := h.ID()
	var fullAddrs []multiaddr.Multiaddr

	for _, addr := range h.Addrs() {
		p2pAddr, err := multiaddr.NewMultiaddr(fmt.Sprintf("/p2p/%s", peerID))
		if err != nil {
			continue
		}
		fullAddrs = append(fullAddrs, addr.Encapsulate(p2pAddr))
	}

	return fullAddrs
}

// GetHostInfo returns peer.AddrInfo for connecting to this host.
func GetHostInfo(h host.Host) peer.AddrInfo {
	return peer.AddrInfo{
		ID:    h.ID(),
		Addrs: h.Addrs(),
	}
}
