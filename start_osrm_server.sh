#!/bin/bash
# This script attempts to start the OSRM server in a Docker container.

# --- Configuration ---
OSRM_IMAGE="osrm/osrm-backend"
OSRM_DATA_DIR="${PWD}/osrm-data"
OSRM_DATA_FILE="data.osrm" # Changed from norway-latest.osrm
HOST_PORT=5001
CONTAINER_PORT=5000

# --- Functions ---
check_docker() {
    echo "Checking if Docker is running..."
    if ! docker info >/dev/null 2>&1; then
        echo "Error: Docker is not running or not accessible."
        echo "Please start Docker and try again."
        exit 1
    fi
    echo "Docker is running."
}

start_osrm() {
    echo "Attempting to start the OSRM server..."

    # Stop and remove any existing container with the same name
    if [ "$(docker ps -a -q -f name=osrm_server)" ]; then
        echo "Stopping and removing existing osrm_server container..."
        docker stop osrm_server
        docker rm osrm_server
    fi

    # Check if the container is already running (after potential stop/remove)
    if [ "$(docker ps -q -f name=osrm_server)" ]; then
        echo "OSRM server is already running."
        return 0
    fi
    
    # Check if the data file exists
    if [ ! -f "${OSRM_DATA_DIR}/${OSRM_DATA_FILE}" ]; then
        echo "Error: OSRM data file not found at '${OSRM_DATA_DIR}/${OSRM_DATA_FILE}'"
        echo "Please ensure the OSRM data is downloaded and in the correct directory."
        return 1
    fi

    echo "Starting OSRM container..."
    docker run -d --name osrm_server \
        -p ${HOST_PORT}:${CONTAINER_PORT} \
        -v "${OSRM_DATA_DIR}:/data" \
        ${OSRM_IMAGE} osrm-routed --algorithm mld "/data/${OSRM_DATA_FILE}"

    # Check if the container started successfully
    if [ $? -eq 0 ]; then
        echo "OSRM container started successfully."
        echo "Waiting a few seconds for the server to initialize..."
        sleep 5 # Give it a moment to start up
        # Verify it's still running
        if [ "$(docker ps -q -f name=osrm_server)" ]; then
            echo "OSRM server is up and running."
            return 0
        else
            echo "Error: OSRM container failed to stay running. Check Docker logs:"
            echo "  docker logs osrm_server"
            return 1
        fi
    else
        echo "Error: Failed to start the OSRM container."
        return 1
    fi
}


# --- Main ---
check_docker
start_osrm
