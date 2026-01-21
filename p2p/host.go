// Package p2p provides libp2p-based peer-to-peer networking for NeuroGrid.
package p2p

import (
	"context"
	"fmt"

	"github.com/libp2p/go-libp2p"
	"github.com/libp2p/go-libp2p/core/host"
	"github.com/libp2p/go-libp2p/core/peer"
	"github.com/multiformats/go-multiaddr"
)

// NewHost creates a new libp2p host with TCP and QUIC transports.
func NewHost(ctx context.Context, listenPort int) (host.Host, error) {
	// Build listen addresses
	tcpAddr := fmt.Sprintf("/ip4/0.0.0.0/tcp/%d", listenPort)
	quicAddr := fmt.Sprintf("/ip4/0.0.0.0/udp/%d/quic-v1", listenPort)

	opts := []libp2p.Option{
		libp2p.ListenAddrStrings(tcpAddr, quicAddr),
		libp2p.EnableHolePunching(),
		libp2p.EnableRelay(),
		libp2p.NATPortMap(),
	}

	h, err := libp2p.New(opts...)
	if err != nil {
		return nil, fmt.Errorf("failed to create libp2p host: %w", err)
	}

	return h, nil
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
