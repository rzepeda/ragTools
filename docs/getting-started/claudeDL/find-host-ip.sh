#!/bin/bash
# find-host-ip.sh - Find the host machine IP from within a VM

echo "🔍 Finding Host IP from VM..."
echo ""

# Common host IPs for different VM configurations
CANDIDATE_IPS=(
    "10.0.2.2"         # VirtualBox NAT (most common)
    "192.168.56.1"     # VirtualBox Host-Only
    "192.168.122.1"    # KVM/QEMU default
    "172.16.0.1"       # VMware NAT
    "192.168.159.1"    # VMware NAT alternative
)

# Try to get gateway IP (often the host)
GATEWAY=$(ip route | grep default | awk '{print $3}' | head -1)
if [ -n "$GATEWAY" ]; then
    echo "📍 Gateway IP (likely host): $GATEWAY"
    CANDIDATE_IPS=("$GATEWAY" "${CANDIDATE_IPS[@]}")
fi

echo ""
echo "Testing candidate IPs for PostgreSQL (port 5432)..."
echo "=================================================="

FOUND_IP=""
for ip in "${CANDIDATE_IPS[@]}"; do
    echo -n "Testing $ip:5432... "
    
    # Test if port is reachable
    if timeout 2 bash -c "cat < /dev/null > /dev/tcp/$ip/5432" 2>/dev/null; then
        echo "✅ SUCCESS"
        FOUND_IP=$ip
        break
    else
        echo "❌ Failed"
    fi
done

echo ""

if [ -n "$FOUND_IP" ]; then
    echo "🎉 Found host at: $FOUND_IP"
    echo ""
    echo "Testing other services on this IP..."
    echo "====================================="
    
    # Test Neo4j Bolt
    echo -n "Neo4j Bolt (7687)... "
    if timeout 2 bash -c "cat < /dev/null > /dev/tcp/$FOUND_IP/7687" 2>/dev/null; then
        echo "✅"
    else
        echo "❌"
    fi
    
    # Test Neo4j HTTP
    echo -n "Neo4j HTTP (7474)... "
    if timeout 2 bash -c "cat < /dev/null > /dev/tcp/$FOUND_IP/7474" 2>/dev/null; then
        echo "✅"
    else
        echo "❌"
    fi
    
    # Test LM Studio
    echo -n "LM Studio (1234)... "
    if timeout 2 bash -c "cat < /dev/null > /dev/tcp/$FOUND_IP/1234" 2>/dev/null; then
        echo "✅"
    else
        echo "❌ (Maybe LM Studio not running?)"
    fi
    
    echo ""
    echo "📝 Update your .env file with:"
    echo "================================"
    echo ""
    echo "HOST_IP=$FOUND_IP"
    echo ""
    echo "DATABASE_URL=postgresql://rag_user:rag_password@${FOUND_IP}:5432/rag_factory"
    echo "DATABASE_TEST_URL=postgresql://rag_user:rag_password@${FOUND_IP}:5432/rag_test"
    echo "NEO4J_URI=bolt://${FOUND_IP}:7687"
    echo "OPENAI_API_BASE=http://${FOUND_IP}:1234/v1"
    echo ""
    
    # Offer to create .env
    if [ -f ".env.example" ]; then
        echo "Would you like to update .env with this IP? (y/n)"
        read -r response
        if [[ "$response" =~ ^[Yy]$ ]]; then
            if [ ! -f ".env" ]; then
                cp .env.example .env
            fi
            
            # Update .env with found IP
            sed -i "s/HOST_IP=.*/HOST_IP=$FOUND_IP/" .env
            sed -i "s/@HOST_IP:/@${FOUND_IP}:/g" .env
            sed -i "s|http://HOST_IP:|http://${FOUND_IP}:|g" .env
            
            echo "✅ Updated .env with HOST_IP=$FOUND_IP"
        fi
    fi
    
else
    echo "❌ Could not find host IP automatically"
    echo ""
    echo "Manual steps:"
    echo "============="
    echo "1. Find your gateway IP:"
    echo "   ip route | grep default"
    echo ""
    echo "2. Or try these common IPs:"
    for ip in "${CANDIDATE_IPS[@]}"; do
        echo "   - $ip"
    done
    echo ""
    echo "3. Test manually:"
    echo "   nc -zv IP_ADDRESS 5432"
    echo ""
    echo "4. Check if PostgreSQL is running on host:"
    echo "   (On host machine) docker compose ps"
    echo ""
    echo "5. Check if port is exposed:"
    echo "   (On host machine) netstat -tlnp | grep 5432"
    echo "   Should show: 0.0.0.0:5432"
    exit 1
fi
