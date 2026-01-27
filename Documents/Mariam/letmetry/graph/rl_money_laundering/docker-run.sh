#!/bin/bash
# Quick start script for Docker Compose setup

set -e

echo "========================================="
echo "RL Money Laundering Detection - Docker Setup"
echo "========================================="
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if docker-compose is available
if ! command -v docker-compose &> /dev/null; then
    echo -e "${YELLOW}Warning: docker-compose not found. Trying 'docker compose' instead...${NC}"
    DOCKER_COMPOSE="docker compose"
else
    DOCKER_COMPOSE="docker-compose"
fi

# Function to display usage
usage() {
    echo "Usage: ./docker-run.sh [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  build       Build Docker images"
    echo "  train       Run training (quick test)"
    echo "  train-full  Run full AMLNet training"
    echo "  jupyter     Start Jupyter Lab"
    echo "  tensorboard Start TensorBoard"
    echo "  shell       Open interactive shell in trainer container"
    echo "  gpu-check   Check GPU availability"
    echo "  logs        Show training logs"
    echo "  stop        Stop all containers"
    echo "  clean       Stop and remove all containers"
    echo ""
    exit 1
}

# Build images
build() {
    echo -e "${GREEN}Building Docker images...${NC}"
    $DOCKER_COMPOSE build
    echo -e "${GREEN}✓ Build complete!${NC}"
}

# Run quick training test
train() {
    echo -e "${GREEN}Starting quick training test...${NC}"
    $DOCKER_COMPOSE run --rm rl-trainer uv run python scripts/train_agent_v2.py --config quick_test
}

# Run full AMLNet training
train_full() {
    echo -e "${GREEN}Starting full AMLNet training...${NC}"
    $DOCKER_COMPOSE run --rm rl-trainer uv run python scripts/train_agent_v2.py --config amlnet_full
}

# Start Jupyter Lab
jupyter() {
    echo -e "${GREEN}Starting Jupyter Lab...${NC}"
    echo "Access at: http://localhost:8888"
    $DOCKER_COMPOSE up jupyter
}

# Start TensorBoard
tensorboard() {
    echo -e "${GREEN}Starting TensorBoard...${NC}"
    echo "Access at: http://localhost:6006"
    $DOCKER_COMPOSE up tensorboard
}

# Open interactive shell
shell() {
    echo -e "${GREEN}Opening interactive shell...${NC}"
    $DOCKER_COMPOSE run --rm rl-trainer /bin/bash
}

# Check GPU
gpu_check() {
    echo -e "${GREEN}Checking GPU availability...${NC}"
    $DOCKER_COMPOSE run --rm rl-trainer nvidia-smi
}

# Show logs
logs() {
    echo -e "${GREEN}Showing training logs...${NC}"
    $DOCKER_COMPOSE logs -f rl-trainer
}

# Stop containers
stop() {
    echo -e "${YELLOW}Stopping containers...${NC}"
    $DOCKER_COMPOSE stop
    echo -e "${GREEN}✓ Containers stopped${NC}"
}

# Clean up
clean() {
    echo -e "${YELLOW}Stopping and removing containers...${NC}"
    $DOCKER_COMPOSE down
    echo -e "${GREEN}✓ Cleanup complete${NC}"
}

# Main command dispatcher
case "${1:-}" in
    build)
        build
        ;;
    train)
        train
        ;;
    train-full)
        train_full
        ;;
    jupyter)
        jupyter
        ;;
    tensorboard)
        tensorboard
        ;;
    shell)
        shell
        ;;
    gpu-check)
        gpu_check
        ;;
    logs)
        logs
        ;;
    stop)
        stop
        ;;
    clean)
        clean
        ;;
    *)
        usage
        ;;
esac
