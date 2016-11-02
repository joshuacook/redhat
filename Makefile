DIRNAME=redhat
default: build up

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

mongo:
	docker exec -it $(DIRNAME)_mongo_1 mongo

postgres:
	docker exec -it $(DIRNAME)_postgres_1 psql postgres postgres 

presentation:
	pandoc -t beamer doc/presentation.md -o presentation.pdf

redis:
	docker exec -it $(DIRNAME)_redis_1 redis-cli

report:
	pandoc --template=doc/template.latex doc/report.md -o doc/report.pdf

rm:
	docker-compose rm

swarm:
	bash bin/digital_ocean_swarm.sh

up:
	docker-compose up

watch_report:
	while true; do kqwait doc/report.md; make report; echo 'next'; done

workers:
	docker-compose scale rq=$(WORKERS)
