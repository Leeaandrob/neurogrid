// Package p2p provides libp2p-based peer-to-peer networking for NeuroGrid.
package p2p

import (
	"context"
	"sync"

	"github.com/libp2p/go-libp2p/core/host"
	"github.com/libp2p/go-libp2p/core/peer"
	"github.com/libp2p/go-libp2p-kad-dht"
	"github.com/libp2p/go-libp2p/p2p/discovery/mdns"
)

const (
	// NeuroGridServiceTag is the mDNS service tag for NeuroGrid discovery.
	NeuroGridServiceTag = "_neurogrid._tcp"
)

// Discovery manages peer discovery via mDNS and DHT.
type Discovery struct {
	host     host.Host
	peerChan chan peer.AddrInfo
	dht      *dht.IpfsDHT
	mdns     mdns.Service
	mu       sync.Mutex
}

// NewDiscovery creates a new Discovery manager for the given host.
func NewDiscovery(h host.Host) *Discovery {
	return &Discovery{
		host:     h,
		peerChan: make(chan peer.AddrInfo, 100),
	}
}

// SetupMDNS initializes mDNS-based local peer discovery.
func (d *Discovery) SetupMDNS() error {
	d.mu.Lock()
	defer d.mu.Unlock()

	// Create mDNS service
	svc := mdns.NewMdnsService(d.host, NeuroGridServiceTag, d)
	if err := svc.Start(); err != nil {
		return err
	}

	d.mdns = svc
	return nil
}

// HandlePeerFound is called by mDNS when a peer is discovered.
// It implements the mdns.Notifee interface.
func (d *Discovery) HandlePeerFound(pi peer.AddrInfo) {
	// Don't notify about ourselves
	if pi.ID == d.host.ID() {
		return
	}

	// Non-blocking send to peer channel
	select {
	case d.peerChan <- pi:
	default:
		// Channel full, drop the notification
	}
}

// SetupDHT initializes the Kademlia DHT for remote peer discovery.
func (d *Discovery) SetupDHT(ctx context.Context) error {
	d.mu.Lock()
	defer d.mu.Unlock()

	// Create DHT in server mode for better connectivity
	kadDHT, err := dht.New(ctx, d.host, dht.Mode(dht.ModeAutoServer))
	if err != nil {
		return err
	}

	// Bootstrap the DHT
	if err := kadDHT.Bootstrap(ctx); err != nil {
		kadDHT.Close()
		return err
	}

	d.dht = kadDHT
	return nil
}

// DHT returns the underlying Kademlia DHT instance.
func (d *Discovery) DHT() *dht.IpfsDHT {
	d.mu.Lock()
	defer d.mu.Unlock()
	return d.dht
}

// PeerChan returns the channel that receives discovered peers.
func (d *Discovery) PeerChan() <-chan peer.AddrInfo {
	return d.peerChan
}

// Close shuts down discovery services.
func (d *Discovery) Close() error {
	d.mu.Lock()
	defer d.mu.Unlock()

	if d.mdns != nil {
		d.mdns.Close()
	}

	if d.dht != nil {
		return d.dht.Close()
	}

	close(d.peerChan)
	return nil
}
