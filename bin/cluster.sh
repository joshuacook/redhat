#!/bin/bash

# configuration
agents="agent1 agent2"

token=$(docker run --rm swarm create)

# Swarm manager machine
echo "Create swarm manager"
docker-machine create \
    -d digitalocean \
    --digitalocean-access-token=$DO_TOKEN \
    --swarm --swarm-master \
    --swarm-discovery token://$token \
    manager

# Swarm agents
for agent in $agents; do
    (
    echo "Creating ${agent}"

    docker-machine create \
        -d digitalocean \
        --digitalocean-access-token=$DO_TOKEN \
        --swarm \
        --swarm-discovery token://$token \
        $agent
    ) &
done
wait

# Information
echo ""
echo "CLUSTER INFORMATION"
echo "discovery token: ${token}"
echo "Environment variables to connect trough docker cli"
docker-machine env --swarm manager
