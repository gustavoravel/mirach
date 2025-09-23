// Mirach - Interactive JavaScript

document.addEventListener('DOMContentLoaded', function() {
    
    // Initialize animations and interactions
    initAnimations();
    initFormEnhancements();
    initCardInteractions();
    initLoadingStates();
    initTooltips();
    
});

function initAnimations() {
    // Animate elements on scroll
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };
    
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.animation = 'fadeInUp 0.6s ease-out forwards';
            }
        });
    }, observerOptions);
    
    // Observe all cards
    document.querySelectorAll('.card').forEach(card => {
        observer.observe(card);
    });
}

function initFormEnhancements() {
    // Enhanced form interactions
    const formControls = document.querySelectorAll('.form-control, .form-select');
    
    formControls.forEach(control => {
        // Add floating label effect
        control.addEventListener('focus', function() {
            this.parentElement.classList.add('focused');
        });
        
        control.addEventListener('blur', function() {
            if (!this.value) {
                this.parentElement.classList.remove('focused');
            }
        });
        
        // Add ripple effect on click
        control.addEventListener('click', function(e) {
            createRipple(e, this);
        });
    });
}

function initCardInteractions() {
    // Enhanced card hover effects
    const cards = document.querySelectorAll('.card');
    
    cards.forEach(card => {
        card.addEventListener('mouseenter', function() {
            this.style.transform = 'translateY(-8px) scale(1.02)';
            this.style.boxShadow = '0 20px 40px rgba(0, 0, 0, 0.15)';
        });
        
        card.addEventListener('mouseleave', function() {
            this.style.transform = 'translateY(0) scale(1)';
            this.style.boxShadow = '0 10px 30px rgba(0, 0, 0, 0.1)';
        });
        
        // Add click ripple effect
        card.addEventListener('click', function(e) {
            if (e.target.tagName !== 'A' && e.target.tagName !== 'BUTTON') {
                createRipple(e, this);
            }
        });
    });
}

function initLoadingStates() {
    // Add loading spinner on form submit without blocking submission
    const forms = document.querySelectorAll('form');
    forms.forEach(form => {
        form.addEventListener('submit', function() {
            const submitBtn = form.querySelector('button[type="submit"]');
            if (submitBtn) {
                showLoading(submitBtn);
            }
        });
    });
}

function initTooltips() {
    // Initialize Bootstrap tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
}

function createRipple(event, element) {
    const ripple = document.createElement('span');
    const rect = element.getBoundingClientRect();
    const size = Math.max(rect.width, rect.height);
    const x = event.clientX - rect.left - size / 2;
    const y = event.clientY - rect.top - size / 2;
    
    ripple.style.width = ripple.style.height = size + 'px';
    ripple.style.left = x + 'px';
    ripple.style.top = y + 'px';
    ripple.classList.add('ripple');
    
    element.appendChild(ripple);
    
    setTimeout(() => {
        ripple.remove();
    }, 600);
}

function showLoading(button) {
    button.dataset.originalText = button.dataset.originalText || button.innerHTML;
    button.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Processando...';
    button.classList.add('loading');
}

// File upload enhancements
function initFileUpload() {
    const fileInputs = document.querySelectorAll('input[type="file"]');
    
    fileInputs.forEach(input => {
        input.addEventListener('change', function(e) {
            const file = e.target.files[0];
            if (file) {
                showFilePreview(file, this);
            }
        });
        
        // Add drag and drop functionality
        input.addEventListener('dragover', function(e) {
            e.preventDefault();
            this.parentElement.classList.add('drag-over');
        });
        
        input.addEventListener('dragleave', function(e) {
            e.preventDefault();
            this.parentElement.classList.remove('drag-over');
        });
        
        input.addEventListener('drop', function(e) {
            e.preventDefault();
            this.parentElement.classList.remove('drag-over');
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                this.files = files;
                showFilePreview(files[0], this);
            }
        });
    });
}

function showFilePreview(file, input) {
    const preview = document.createElement('div');
    preview.className = 'file-preview';
    preview.innerHTML = `
        <div class="file-info">
            <i class="fas fa-file-excel text-success"></i>
            <span class="file-name">${file.name}</span>
            <span class="file-size">${formatFileSize(file.size)}</span>
        </div>
        <div class="file-progress">
            <div class="progress-bar"></div>
        </div>
    `;
    
    input.parentElement.appendChild(preview);
    
    // Animate progress bar
    setTimeout(() => {
        const progressBar = preview.querySelector('.progress-bar');
        progressBar.style.width = '100%';
    }, 100);
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

// Notification system
function showNotification(message, type = 'info', duration = 5000) {
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.innerHTML = `
        <div class="notification-content">
            <i class="fas fa-${getNotificationIcon(type)}"></i>
            <span>${message}</span>
            <button class="notification-close">&times;</button>
        </div>
    `;
    
    document.body.appendChild(notification);
    
    // Animate in
    setTimeout(() => {
        notification.classList.add('show');
    }, 100);
    
    // Auto remove
    setTimeout(() => {
        removeNotification(notification);
    }, duration);
    
    // Manual close
    notification.querySelector('.notification-close').addEventListener('click', () => {
        removeNotification(notification);
    });
}

function removeNotification(notification) {
    notification.classList.remove('show');
    setTimeout(() => {
        notification.remove();
    }, 300);
}

function getNotificationIcon(type) {
    const icons = {
        'success': 'check-circle',
        'error': 'exclamation-circle',
        'warning': 'exclamation-triangle',
        'info': 'info-circle'
    };
    return icons[type] || 'info-circle';
}

// Chart animations
function animateChart(chartElement) {
    const bars = chartElement.querySelectorAll('.chart-bar');
    bars.forEach((bar, index) => {
        setTimeout(() => {
            bar.style.transform = 'scaleY(1)';
            bar.style.opacity = '1';
        }, index * 100);
    });
}

// Search functionality
function initSearch() {
    const searchInputs = document.querySelectorAll('.search-input');
    
    searchInputs.forEach(input => {
        input.addEventListener('input', function(e) {
            const query = e.target.value.toLowerCase();
            const items = document.querySelectorAll('.searchable-item');
            
            items.forEach(item => {
                const text = item.textContent.toLowerCase();
                if (text.includes(query)) {
                    item.style.display = 'block';
                    item.style.animation = 'fadeInUp 0.3s ease-out';
                } else {
                    item.style.display = 'none';
                }
            });
        });
    });
}

// Initialize all features
document.addEventListener('DOMContentLoaded', function() {
    initFileUpload();
    initSearch();
});

// Add CSS for ripple effect
const style = document.createElement('style');
style.textContent = `
    .ripple {
        position: absolute;
        border-radius: 50%;
        background: rgba(255, 255, 255, 0.6);
        transform: scale(0);
        animation: ripple-animation 0.6s linear;
        pointer-events: none;
    }
    
    @keyframes ripple-animation {
        to {
            transform: scale(4);
            opacity: 0;
        }
    }
    
    .file-preview {
        margin-top: 10px;
        padding: 15px;
        background: rgba(255, 255, 255, 0.9);
        border-radius: 12px;
        border: 2px dashed #667eea;
        animation: fadeInUp 0.3s ease-out;
    }
    
    .file-info {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 10px;
    }
    
    .file-name {
        font-weight: 600;
        color: #495057;
    }
    
    .file-size {
        color: #6c757d;
        font-size: 0.9rem;
    }
    
    .file-progress {
        height: 4px;
        background: rgba(102, 126, 234, 0.2);
        border-radius: 2px;
        overflow: hidden;
    }
    
    .progress-bar {
        height: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        width: 0%;
        transition: width 0.3s ease;
    }
    
    .drag-over {
        border-color: #667eea !important;
        background: rgba(102, 126, 234, 0.1) !important;
    }
    
    .notification {
        position: fixed;
        top: 20px;
        right: 20px;
        z-index: 9999;
        background: white;
        border-radius: 12px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
        transform: translateX(400px);
        transition: transform 0.3s ease;
        max-width: 400px;
    }
    
    .notification.show {
        transform: translateX(0);
    }
    
    .notification-content {
        padding: 15px 20px;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    
    .notification-close {
        background: none;
        border: none;
        font-size: 20px;
        cursor: pointer;
        color: #6c757d;
        margin-left: auto;
    }
    
    .notification-success {
        border-left: 4px solid #4facfe;
    }
    
    .notification-error {
        border-left: 4px solid #fa709a;
    }
    
    .notification-warning {
        border-left: 4px solid #43e97b;
    }
    
    .notification-info {
        border-left: 4px solid #667eea;
    }
`;
document.head.appendChild(style);
