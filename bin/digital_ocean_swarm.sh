#!/bin/bash
date

function create_docker_node {
    docker-machine create \
      --driver=digitalocean \
      --digitalocean-access-token=$DO_TOKEN \
      --digitalocean-size=2gb \
      --digitalocean-region=$DO_REGION \
      --digitalocean-private-networking=true \
      --digitalocean-ssh-user=$DO_USER \
      --digitalocean-ssh-port=22 \
      --digitalocean-ssh-key-fingerprint=$DO_SSH_KEY_FINGERPRINT \
      --digitalocean-image=docker \
      --swarm \
      --swarm-discovery consul://${kvip}:8500 \
      --engine-opt "cluster-store consul://${kvip}:8500" \
      --engine-opt "cluster-advertise eth1:2376" \
        docker-swarm-agent-$1
}

export -f create_docker_node

function join_swarm_agent {
    eval $(docker-machine env docker-swarm-agent-$1)
    docker run -d swarm join --addr=$(docker-machine ip docker-swarm-agent-$1):2376 token://$2
}

export -f join_swarm_agent

if [ -z "$1" ]
  then
        echo 'create_master'
        echo 'create_nodes #NUMBER_OF_NODES#'
        echo 'destroy_all_nodes'
        echo 'destroy'
else
    NUMBER_OF_NODES=$2
fi

case $1 in
    create_master )

        docker-machine create \
          --driver=digitalocean \
          --digitalocean-access-token=$DO_TOKEN \
          --digitalocean-size=512mb \
          --digitalocean-region=$DO_USER \
          --digitalocean-region=$DO_REGION \
          --digitalocean-private-networking=true \
          --digitalocean-ssh-key-fingerprint=$DO_SSH_KEY_FINGERPRINT \
          --digitalocean-image=docker \
            docker-swarm-kv-store

        docker $(docker-machine config docker-swarm-kv-store) run -d \
        --net=host progrium/consul --server -bootstrap-expect 1

        kvip=$(docker-machine ip docker-swarm-kv-store)

        docker-machine create \
          --driver=digitalocean \
          --digitalocean-access-token=$DO_TOKEN \
          --digitalocean-size=2gb \
          --digitalocean-region=$DO_REGION \
          --digitalocean-private-networking=true \
          --swarm \
          --swarm-master \
          --swarm-discovery consul://${kvip}:8500 \
          --engine-opt "cluster-store consul://${kvip}:8500" \
          --engine-opt "cluster-advertise eth1:2376" \
          --digitalocean-ssh-key-fingerprint=$DO_SSH_KEY_FINGERPRINT \
          --digitalocean-image=docker \
            docker-swarm-master
    ;;
    create_nodes)
        if [ -z "$2" ]
          then
            NUMBER_OF_NODES=2
        else
            NUMBER_OF_NODES=$2
        fi

        parallel --gnu create_docker_node ::: $(seq $NUMBER_OF_NODES)

        eval $(docker-machine env --swarm docker-swarm-master)
        docker info
    ;;
    destroy_all_nodes)
        yes | docker-machine rm $(docker-machine ls -q | grep docker-swarm-agent)
    ;;  
    destroy )
    yes | docker-machine rm $(docker-machine ls -q | grep docker-swarm)
    ;;
    src )
    cd
    docker-machine scp -r $(pwd) docker-swarm-master:
    ;;
esac

date


