DIRNAME=redhat
default: build up

bash_ipython:
	docker exec -it $(DIRNAME)_jupyter_1 ipython
bash_jupyter:
	docker exec -it $(DIRNAME)_jupyter_1 /bin/bash

bash_rq:
	docker exec -it $(DIRNAME)_rq_1 /bin/bash

bash_webserver:
	docker exec -it $(DIRNAME)_webserver_1 /bin/bash

build:
	docker-compose build

down:
	docker-compose down

interface:
	docker exec -it $(DIRNAME)_webserver_1 python -m lib.app.interface

postgres:
	docker exec -it $(DIRNAME)_postgres_1 psql postgres postgres 

presentation:
	pandoc -t beamer doc/presentation.md -o doc/presentation.pdf

redis:
	docker exec -it $(DIRNAME)_redis_1 redis-cli

report:
	docker run -v $(shell pwd):/source joshuacook/pandoc --template=doc/template.latex doc/report.md -o doc/report.pdf

rm:
	docker-compose rm

swarm:
	bash bin/digital_ocean_swarm.sh

swarm_local:
	$(shell docker-machine env -u)

swarm_token:
	echo SWARM_CLUSTER_TOKEN=$(shell docker run swarm create) > swarm_cluster_token.txt
	echo 'Run $ . swarm_cluster_token.txt'

up:
	docker-compose up

watch_report:
	while true; do kqwait doc/report.md; make report; echo 'next'; done

workers:
	docker-compose scale rq=$(WORKERS)
