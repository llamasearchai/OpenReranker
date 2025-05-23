#!/bin/bash

# OpenReranker Development Environment Setup Script
# This script sets up a complete development environment for OpenReranker

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

# Function to check Python version
check_python_version() {
    print_status "Checking Python version..."
    
    if command_exists python3; then
        PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
        PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
        PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)
        
        if [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -ge 9 ]; then
            print_success "Python $PYTHON_VERSION found"
            PYTHON_CMD="python3"
        else
            print_error "Python 3.9+ required, found $PYTHON_VERSION"
            exit 1
        fi
    else
        print_error "Python 3 not found. Please install Python 3.9+"
        exit 1
    fi
}

# Function to create virtual environment
create_venv() {
    print_status "Creating virtual environment..."
    
    if [ -d "venv" ]; then
        print_warning "Virtual environment already exists. Removing..."
        rm -rf venv
    fi
    
    $PYTHON_CMD -m venv venv
    print_success "Virtual environment created"
}

# Function to activate virtual environment
activate_venv() {
    print_status "Activating virtual environment..."
    
    if [ -f "venv/bin/activate" ]; then
        source venv/bin/activate
        print_success "Virtual environment activated"
    else
        print_error "Virtual environment activation script not found"
        exit 1
    fi
}

# Function to upgrade pip
upgrade_pip() {
    print_status "Upgrading pip..."
    pip install --upgrade pip
    print_success "Pip upgraded"
}

# Function to install dependencies
install_dependencies() {
    print_status "Installing dependencies..."
    
    # Install build tools
    pip install build wheel setuptools
    
    # Install the package in development mode
    pip install -e .
    
    # Install development dependencies
    pip install pytest>=7.0.0
    pip install pytest-cov>=4.0.0
    pip install pytest-mock>=3.10.0
    pip install pytest-asyncio>=0.21.0
    pip install httpx>=0.24.0
    
    # Install code quality tools
    pip install black>=23.0.0
    pip install isort>=5.12.0
    pip install flake8>=6.0.0
    pip install mypy>=1.4.0
    
    # Install security tools
    pip install bandit>=1.7.0
    pip install safety>=2.3.0
    
    # Install tox for testing across Python versions
    pip install tox>=4.0.0
    
    # Install optional dependencies for integrations (if available)
    pip install dspy-ai || print_warning "DSPy not available - integration tests will be skipped"
    pip install langchain-core || print_warning "LangChain not available - integration tests will be skipped"
    
    print_success "Dependencies installed"
}

# Function to run initial tests
run_initial_tests() {
    print_status "Running initial tests to verify setup..."
    
    # Run a quick test to make sure everything works
    python -c "import open_reranker; print('✅ Package import successful')"
    
    # Run basic tests
    pytest tests/ -v --tb=short -x
    
    print_success "Initial tests passed"
}

# Function to create development configuration
create_dev_config() {
    print_status "Creating development configuration..."
    
    # Create .env file for development
    cat > .env << EOF
# Development environment variables
OPEN_RERANKER_DEBUG=true
OPEN_RERANKER_HOST=0.0.0.0
OPEN_RERANKER_PORT=8000
OPEN_RERANKER_USE_MLX=false
EOF
    
    # Create pytest configuration
    cat > pytest.ini << EOF
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = -v --tb=short
markers =
    slow: marks tests as slow (deselect with '-m "not slow"')
    integration: marks tests as integration tests
EOF
    
    # Create mypy configuration
    cat > mypy.ini << EOF
[mypy]
python_version = 3.9
warn_return_any = True
warn_unused_configs = True
disallow_untyped_defs = True
disallow_incomplete_defs = True
check_untyped_defs = True
disallow_untyped_decorators = True
no_implicit_optional = True
warn_redundant_casts = True
warn_unused_ignores = True
warn_no_return = True
warn_unreachable = True
strict_equality = True

[mypy-tests.*]
disallow_untyped_defs = False
disallow_incomplete_defs = False

[mypy-transformers.*]
ignore_missing_imports = True

[mypy-torch.*]
ignore_missing_imports = True

[mypy-dspy.*]
ignore_missing_imports = True

[mypy-langchain_core.*]
ignore_missing_imports = True
EOF
    
    print_success "Development configuration created"
}

# Function to display final instructions
display_instructions() {
    print_success "Development environment setup complete!"
    echo ""
    echo "To get started:"
    echo "1. Activate the virtual environment:"
    echo "   source venv/bin/activate"
    echo ""
    echo "2. Run the development server:"
    echo "   make run"
    echo "   # or"
    echo "   uvicorn open_reranker.main:app --reload"
    echo ""
    echo "3. Run tests:"
    echo "   make test"
    echo "   # or"
    echo "   pytest tests/"
    echo ""
    echo "4. Format code:"
    echo "   make format"
    echo ""
    echo "5. Run all checks:"
    echo "   make dev-check"
    echo ""
    echo "Available make targets:"
    echo "   make help    # Show all available commands"
    echo ""
    echo "Happy coding! 🚀"
}

# Main setup function
main() {
    echo "🔧 Setting up OpenReranker development environment..."
    echo ""
    
    check_python_version
    create_venv
    activate_venv
    upgrade_pip
    install_dependencies
    create_dev_config
    run_initial_tests
    
    echo ""
    display_instructions
}

# Run main function
main "$@" 