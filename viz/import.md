# build the graph database

please copy and paste these cmd in bash and excute them one by one

1. pull neo4j docker image from docker hub
    ```bash
    docker pull neo4j:4.4.16-community
    ```
2. run following cmd at cancerGraph dir (~~not cancerGraph/viz dir~~) to start the database
    ```bash 
    vizDir=$(pwd)

    docker run -itd\
        --publish=7474:7474 --publish=7687:7687 \
        --env NEO4J_AUTH=neo4j/admin \
        -e NEO4JLABS_PLUGINS=\[\"apoc-core\"\] \
        -e NEO4J.dbms.security.procedures.unrestricted=apoc.\\\* \
        -e NEO4J_apoc_export_file_enabled=true \
        -e NEO4J_apoc_import_file_enabled=true \
        -e NEO4J_apoc_import_file_use__neo4j__config=true \
        -e NEO4J_db_import_csv_buffer__size=1G \
        --volume="$vizDir"/data/graphData/:/import \
        --volume="$vizDir"/viz/tmpData:/data \
        --name=cancerGraphDB \
        --user="$(id -u):$(id -g)" \
        neo4j:4.4.16-community
    ```

3. wait ~8s for the database server to start and then import graphData
    ```bash
    docker exec -it cancerGraphDB \
        neo4j-admin import \
            --delimiter="\t" \
            --array-delimiter="," \
            --trim-strings=true \
            --processors=16 \
            --database=neo4j \
            --skip-bad-relationships=true \
            --skip-duplicate-nodes=true \
            --read-buffer-size=200m \
            --force \
            --nodes=Concept=/import/concepts_header.tsv,/import/concepts.tsv \
            --nodes=Paper=/import/papers_header.tsv,/import/papers.tsv \
            --relationships=Link=/import/triples_header.tsv,/import/triples_no_duplicate.tsv 
    ```

4. create some index to accelerate search operation
    
    start cypher-shell first
    ```bash
    docker exec -it cancerGraphDB cypher-shell -u neo4j -p admin
    ```
    then run in the shell
    ```sql
    CREATE INDEX ON :Concept (cid);
    CREATE INDEX ON :Paper (pid);
    :exit;
    ```