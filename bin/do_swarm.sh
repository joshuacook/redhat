#!/bin/bash

function create_manager {
    docker-machine create \
      --driver=digitalocean \
      --digitalocean-access-token=${DO_TOKEN} \
      --digitalocean-size=2gb \
      --digitalocean-region=${DO_REGION} \
      --swarm --swarm-master \
      --swarm-discovery token://${SWARM_CLUSTER_TOKEN} \
      swarm-manager
}

export -f create_manager

function create_worker {
    docker-machine create \
      --driver=digitalocean \
      --digitalocean-access-token=${DO_TOKEN} \
      --digitalocean-size=2gb \
      --digitalocean-region=${DO_REGION} \
      --swarm \
      --swarm-discovery token://${SWARM_CLUSTER_TOKEN} \
      swarm-worker-$1
}

export -f create_worker

function destroy {
    case $1 in 
	manager)
        yes | docker-machine rm swarm-manager
        ;;
	worker)
        yes | docker-machine rm swarm-worker-$2
        ;;
    esac
}

export -f destroy

if [ -z "$1" ]
  then
        echo 'create manager'
        echo '       worker #NUMBER#'
        echo 'destroy manager'
        echo '        worker #NUMBER'
fi

case $1 in
    create)
    case $2 in
        manager)
		create_manager
        ;;
        worker)
		create_worker $3
        ;;
    esac
    ;;
    destroy)
    destroy $2 $3 
    ;;
esac
