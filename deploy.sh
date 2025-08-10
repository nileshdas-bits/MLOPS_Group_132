#!/bin/bash

# MLOps Pipeline Deployment Script
# This script handles the complete deployment of the Iris classification MLOps pipeline

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    local missing_deps=()
    
    if ! command_exists python3; then
        missing_deps+=("python3")
    fi
    
    if ! command_exists pip; then
        missing_deps+=("pip")
    fi
    
    if ! command_exists docker; then
        missing_deps+=("docker")
    fi
    
    if ! command_exists git; then
        missing_deps+=("git")
    fi
    
    if [ ${#missing_deps[@]} -ne 0 ]; then
        print_error "Missing dependencies: ${missing_deps[*]}"
        print_error "Please install the missing dependencies and try again."
        exit 1
    fi
    
    print_success "All prerequisites are installed"
}

# Function to install Python dependencies
install_dependencies() {
    print_status "Installing Python dependencies..."
    
    if [ ! -f "requirements.txt" ]; then
        print_error "requirements.txt not found"
        exit 1
    fi
    
    pip install -r requirements.txt
    print_success "Python dependencies installed"
}

# Function to train models
train_models() {
    print_status "Training models..."
    
    if [ ! -f "src/models/train.py" ]; then
        print_error "Training script not found"
        exit 1
    fi
    
    python src/models/train.py
    print_success "Models trained successfully"
}

# Function to run tests
run_tests() {
    print_status "Running tests..."
    
    if command_exists pytest; then
        python -m pytest tests/ -v
        print_success "Tests completed"
    else
        print_warning "pytest not found, skipping tests"
    fi
}

# Function to build Docker image
build_docker() {
    print_status "Building Docker image..."
    
    if [ ! -f "Dockerfile" ]; then
        print_error "Dockerfile not found"
        exit 1
    fi
    
    docker build -t iris-mlops .
    print_success "Docker image built successfully"
}

# Function to run Docker container
run_docker() {
    print_status "Running Docker container..."
    
    # Stop existing container if running
    docker stop iris-mlops-container 2>/dev/null || true
    docker rm iris-mlops-container 2>/dev/null || true
    
    # Run new container
    docker run -d \
        --name iris-mlops-container \
        -p 8000:8000 \
        -v "$(pwd)/logs:/app/logs" \
        -v "$(pwd)/mlruns:/app/mlruns" \
        iris-mlops
    
    print_success "Docker container started"
    print_status "API available at http://localhost:8000"
    print_status "API documentation at http://localhost:8000/docs"
}

# Function to run with docker-compose
run_docker_compose() {
    print_status "Running with docker-compose..."
    
    if [ ! -f "docker-compose.yml" ]; then
        print_error "docker-compose.yml not found"
        exit 1
    fi
    
    docker-compose up -d --build
    print_success "Services started with docker-compose"
    print_status "API available at http://localhost:8000"
    print_status "Prometheus available at http://localhost:9090"
    print_status "Grafana available at http://localhost:3000"
}

# Function to test API
test_api() {
    print_status "Testing API..."
    
    # Wait for API to be ready
    sleep 10
    
    # Test health endpoint
    if curl -f http://localhost:8000/ >/dev/null 2>&1; then
        print_success "API health check passed"
    else
        print_error "API health check failed"
        exit 1
    fi
    
    # Test prediction endpoint
    local test_data='{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}'
    if curl -X POST "http://localhost:8000/predict" \
        -H "Content-Type: application/json" \
        -d "$test_data" >/dev/null 2>&1; then
        print_success "API prediction test passed"
    else
        print_warning "API prediction test failed (model might not be loaded)"
    fi
}

# Function to show status
show_status() {
    print_status "Checking service status..."
    
    echo "=== Docker Containers ==="
    docker ps --filter "name=iris-mlops" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
    
    echo -e "\n=== API Endpoints ==="
    echo "Health Check: http://localhost:8000/"
    echo "API Docs: http://localhost:8000/docs"
    echo "Metrics: http://localhost:8000/metrics"
    echo "Logs: http://localhost:8000/logs"
    
    if docker-compose ps | grep -q "prometheus"; then
        echo -e "\n=== Monitoring ==="
        echo "Prometheus: http://localhost:9090"
        echo "Grafana: http://localhost:3000 (admin/admin)"
    fi
}

# Function to stop services
stop_services() {
    print_status "Stopping services..."
    
    docker-compose down 2>/dev/null || true
    docker stop iris-mlops-container 2>/dev/null || true
    docker rm iris-mlops-container 2>/dev/null || true
    
    print_success "Services stopped"
}

# Function to show help
show_help() {
    echo "MLOps Pipeline Deployment Script"
    echo ""
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  install     Install dependencies and train models"
    echo "  test        Run tests"
    echo "  build       Build Docker image"
    echo "  run         Run with Docker"
    echo "  compose     Run with docker-compose (includes monitoring)"
    echo "  deploy      Full deployment (install + build + run)"
    echo "  status      Show service status"
    echo "  stop        Stop all services"
    echo "  help        Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 deploy          # Full deployment"
    echo "  $0 compose         # Run with monitoring"
    echo "  $0 status          # Check status"
}

# Main script logic
case "${1:-help}" in
    "install")
        check_prerequisites
        install_dependencies
        train_models
        ;;
    "test")
        run_tests
        ;;
    "build")
        build_docker
        ;;
    "run")
        build_docker
        run_docker
        test_api
        ;;
    "compose")
        build_docker
        run_docker_compose
        test_api
        ;;
    "deploy")
        check_prerequisites
        install_dependencies
        train_models
        run_tests
        build_docker
        run_docker_compose
        test_api
        show_status
        ;;
    "status")
        show_status
        ;;
    "stop")
        stop_services
        ;;
    "help"|*)
        show_help
        ;;
esac 