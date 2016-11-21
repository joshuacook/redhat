% Scalable Computing for Individuals
% Joshua Cook
% November 2, 2016

## Problem: Medium-Sized Data

- Kaggle Problems; Datasets from UCI
- Small enough to work with using standard database tools (Postgres, Mongo)
- Large enough to be unwieldy; feature engineering and training is extremely slow
- Advantage of working as an individual can be lost (creativity, rapid innovation)
- Especially, difficulties in using Jupyter with medium to large data sets

## Solution: Infrastructure as Code

Use `docker` and `docker-compose` to define a multi-container system for processing data.

Considering Docker best-practice, one process per container, our system uses the following container types:

Jupyter
:   primary interface to system

Postgres
:   database

Redis
:   memory cache  

Webserver
:   basic webserver designed for monitoring worker health

Worker
:   dedicated python processor

------------------

![Infrastructure](assets/img/infrastructure.png)

------------------

## `docker-compose.yml`

\begin{figure}
   \includegraphics[width=0.475\textwidth]{assets/img/redhat-docker-compose-yml-1.png}
   \hfill
   \includegraphics[width=0.475\textwidth]{assets/img/redhat-docker-compose-yml-2.png}
\end{figure}

## Controlling the System

### shell to containers

```bash
bash_jupyter:
    docker exec -it $(DIRNAME)_jupyter_1 /bin/bash

bash_rq:
    docker exec -it $(DIRNAME)_rq_1 /bin/bash

bash_webserver:
    docker exec -it $(DIRNAME)_webserver_1 /bin/bash

postgres:
    docker exec -it $(DIRNAME)_postgres_1 psql postgres postgres

redis:
    docker exec -it $(DIRNAME)_redis_1 redis-cli
```

## Controlling the System

### build and maintain system

```bash
build:
	docker-compose build

down:
	docker-compose down

rm:
	docker-compose rm

swarm:
	bash bin/digital_ocean_swarm.sh

up:
	docker-compose up

workers:
	docker-compose scale rq=$(WORKERS)
```

## Launching the System

Launch triggers in postgres image

## Queueing Tasks
