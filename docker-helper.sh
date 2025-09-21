#!/bin/bash
# Docker Compose Helper Script for Dev Environment
# This script helps resolve common permission and Docker issues

set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
DOCKER_COMPOSE_FILE="$PROJECT_DIR/docker-compose.yml"
DOCKER_COMPOSE_DEV_FILE="$PROJECT_DIR/docker-compose.dev.yml"

# Enhanced Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
WHITE='\033[1;37m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Emoji support
ROCKET="🚀"
GEAR="⚙️"
CHECK="✅"
CROSS="❌"
WARNING="⚠️"
INFO="ℹ️"
DOCKER="🐳"
SUCCESS="🎉"

log_info() {
    echo -e "${BLUE}${INFO}${NC} ${CYAN}$1${NC}"
}

log_warn() {
    echo -e "${YELLOW}${WARNING}${NC} ${YELLOW}$1${NC}"
}

log_error() {
    echo -e "${RED}${CROSS}${NC} ${RED}$1${NC}"
}

log_success() {
    echo -e "${GREEN}${CHECK}${NC} ${GREEN}$1${NC}"
}

log_docker() {
    echo -e "${BLUE}${DOCKER}${NC} ${CYAN}$1${NC}"
}

log_rocket() {
    echo -e "${MAGENTA}${ROCKET}${NC} ${WHITE}$1${NC}"
}

# Print banner
print_banner() {
    echo -e "${CYAN}${BOLD}"
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║                    Docker Dev Environment                    ║"
    echo "║                      Helper Script                           ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

# Print service status with colors
print_service_status() {
    local service=$1
    local status=$2
    local port=$3

    case $status in
        "running"|"healthy")
            echo -e "${GREEN}${CHECK}${NC} ${WHITE}${service}${NC} ${GREEN}running${NC} on port ${CYAN}${port}${NC"
            ;;
        "starting"|"unhealthy")
            echo -e "${YELLOW}${GEAR}${NC} ${WHITE}${service}${NC} ${YELLOW}starting${NC} on port ${CYAN}${port}${NC"
            ;;
        "exited"|"stopped")
            echo -e "${RED}${CROSS}${NC} ${WHITE}${service}${NC} ${RED}stopped${NC"
            ;;
        *)
            echo -e "${YELLOW}${WARNING}${NC} ${WHITE}${service}${NC} ${YELLOW}unknown${NC} (${status})"
            ;;
    esac
}

# Enhanced status display
show_enhanced_status() {
    log_docker "Service Status Overview"
    echo -e "${CYAN}────────────────────────────────────────${NC}"

    # Check each service
    services=("elasticsearch:9200" "kibana:5601" "redis:6379" "dev-container:2222")

    for service_info in "${services[@]}"; do
        service=$(echo $service_info | cut -d: -f1)
        port=$(echo $service_info | cut -d: -f2)

        if docker-compose -f "$DOCKER_COMPOSE_FILE" ps "$service" 2>/dev/null | grep -q "Up"; then
            container_status=$(docker-compose -f "$DOCKER_COMPOSE_FILE" ps "$service" 2>/dev/null | grep "$service" | awk '{print $4}')
            print_service_status "$service" "$container_status" "$port"
        else
            print_service_status "$service" "stopped" "$port"
        fi
    done

    echo -e "${CYAN}────────────────────────────────────────${NC}"
    log_info "Access URLs:"
    echo -e "  ${WHITE}Elasticsearch:${NC} http://localhost:9200"
    echo -e "  ${WHITE}Kibana:${NC}        http://localhost:5601"
    echo -e "  ${WHITE}Redis:${NC}         localhost:6379"
    echo -e "  ${WHITE}Dev Container:${NC} ssh://localhost:2222"
}

# Check if Docker is running
check_docker() {
    if ! docker info >/dev/null 2>&1; then
        log_error "Docker is not running or not accessible"
        log_info "Please start Docker Desktop or Docker service"
        exit 1
    fi
}

# Fix file permissions
fix_permissions() {
    log_info "Fixing file permissions..."
    # Run permission fix in the dev container
    if docker-compose -f "$DOCKER_COMPOSE_FILE" ps | grep -q "dev-container.*Up"; then
        docker-compose -f "$DOCKER_COMPOSE_FILE" exec dev-container sudo chown -R vscode:vscode /workspaces 2>/dev/null || true
        docker-compose -f "$DOCKER_COMPOSE_FILE" exec dev-container sudo chmod -R 755 /workspaces 2>/dev/null || true
        log_success "File permissions fixed in dev container"
    else
        log_warn "Dev container is not running. Start it first with './docker-helper.sh start'"
    fi
}

# Clean up Docker resources
clean_docker() {
    log_warn "Cleaning up Docker resources..."
    log_warn "This will remove all containers, volumes, and networks"

    read -p "Are you sure? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        log_docker "Removing containers and volumes..."
        docker-compose -f "$DOCKER_COMPOSE_FILE" down -v --remove-orphans 2>/dev/null || true

        log_docker "Pruning unused Docker resources..."
        docker system prune -f >/dev/null 2>&1 || true

        log_success "Cleanup completed!"
    else
        log_info "Cleanup cancelled"
    fi
}

# Start all services
start_services() {
    check_docker
    fix_permissions

    log_docker "Starting all services..."
    echo -e "${YELLOW}This may take a few minutes for Elasticsearch to initialize...${NC}"

    if docker-compose -f "$DOCKER_COMPOSE_FILE" up -d; then
        log_success "Services started successfully!"
        echo -e "${CYAN}Waiting for services to be healthy...${NC}"

        # Show progress
        for i in {1..10}; do
            echo -n -e "${BLUE}Progress: ["
            for j in $(seq 1 $i); do echo -n "="; done
            for j in $(seq $i 9); do echo -n " "; done
            echo -n -e "] $((i*10))%${NC}\r"
            sleep 3
        done
        echo ""

        show_enhanced_status
        log_rocket "Environment is ready! Happy coding! 🎉"
    else
        log_error "Failed to start services"
        exit 1
    fi
}

# Stop all services
stop_services() {
    log_docker "Stopping all services..."
    if docker-compose -f "$DOCKER_COMPOSE_FILE" down; then
        log_success "All services stopped"
    else
        log_error "Failed to stop services"
    fi
}

# Show service status
show_status() {
    log_docker "Service Status:"
    docker-compose -f "$DOCKER_COMPOSE_FILE" ps

    log_docker "Service Health:"
    echo "Elasticsearch: $(curl -s http://localhost:9200/_cluster/health | jq -r '.status' 2>/dev/null || echo 'Not accessible')"
    echo "Kibana: $(curl -s http://localhost:5601/api/status | jq -r '.status.overall.level' 2>/dev/null || echo 'Not accessible')"
    echo "Redis: $(redis-cli -p 6379 ping 2>/dev/null || echo 'Not accessible')"
}

# Show logs
show_logs() {
    service="${1:-all}"
    if [ "$service" = "all" ]; then
        docker-compose -f "$DOCKER_COMPOSE_FILE" logs --tail=50
    else
        docker-compose -f "$DOCKER_COMPOSE_FILE" logs --tail=50 "$service"
    fi
}

# Open interactive shell in dev container
open_shell() {
    if docker-compose -f "$DOCKER_COMPOSE_FILE" ps | grep -q "dev-container.*Up"; then
        log_docker "Opening shell in dev container..."
        docker-compose -f "$DOCKER_COMPOSE_FILE" exec dev-container /bin/bash
    else
        log_error "Dev container is not running. Start it first with './docker-helper.sh start'"
    fi
}

# Enhanced help with colors
show_help() {
    print_banner
    echo -e "${WHITE}${BOLD}Usage:${NC} ${CYAN}$0${NC} ${YELLOW}[command]${NC}"
    echo ""
    echo -e "${WHITE}${BOLD}Commands:${NC}"
    echo -e "  ${GREEN}start${NC}       ${WHITE}Start all services with progress indicator${NC}"
    echo -e "  ${RED}stop${NC}        ${WHITE}Stop all services gracefully${NC}"
    echo -e "  ${YELLOW}restart${NC}     ${WHITE}Restart all services${NC}"
    echo -e "  ${BLUE}status${NC}      ${WHITE}Show enhanced service status with colors${NC}"
    echo -e "  ${CYAN}logs${NC}        ${WHITE}Show all service logs${NC}"
    echo -e "  ${CYAN}logs${NC} ${YELLOW}<svc>${NC}  ${WHITE}Show logs for specific service${NC}"
    echo -e "  ${MAGENTA}clean${NC}       ${WHITE}Clean up containers, volumes, and networks${NC}"
    echo -e "  ${GREEN}fix-perms${NC}   ${WHITE}Fix file permissions${NC}"
    echo -e "  ${BLUE}shell${NC}       ${WHITE}Open interactive shell in dev container${NC}"
    echo -e "  ${WHITE}help${NC}        ${WHITE}Show this colorful help${NC}"
    echo ""
    echo -e "${WHITE}${BOLD}Available Services:${NC}"
    echo -e "  ${DOCKER} ${WHITE}elasticsearch${NC} (port 9200)"
    echo -e "  ${DOCKER} ${WHITE}kibana${NC}        (port 5601)"
    echo -e "  ${DOCKER} ${WHITE}redis${NC}         (port 6379)"
    echo -e "  ${DOCKER} ${WHITE}dev-container${NC} (port 2222)"
    echo ""
    echo -e "${WHITE}${BOLD}Examples:${NC}"
    echo -e "  ${CYAN}$0 start${NC}"
    echo -e "  ${CYAN}$0 logs kibana${NC}"
    echo -e "  ${CYAN}$0 status${NC}"
    echo -e "  ${CYAN}$0 shell${NC}"
    echo ""
    echo -e "${YELLOW}${BOLD}Tips:${NC}"
    echo -e "  ${INFO} Use ${CYAN}./docker-helper.sh${NC} for easy access"
    echo -e "  ${WARNING} Run ${YELLOW}fix-perms${NC} if you encounter permission issues"
    echo -e "  ${ROCKET} Services auto-restart unless manually stopped"
}

# Main logic
main() {
    case "${1:-help}" in
        start)
            start_services
            ;;
        stop)
            stop_services
            ;;
        restart)
            log_rocket "Restarting all services..."
            stop_services
            sleep 2
            start_services
            ;;
        status)
            print_banner
            show_status
            ;;
        logs)
            log_docker "Showing logs for ${2:-all services}..."
            show_logs "$2"
            ;;
        clean)
            print_banner
            clean_docker
            ;;
        fix-perms)
            log_docker "Fixing permissions..."
            fix_permissions
            ;;
        shell)
            open_shell
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            log_error "Unknown command: $1"
            echo ""
            show_help
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"