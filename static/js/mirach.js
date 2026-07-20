// Mirach - Interactive JavaScript

document.addEventListener('DOMContentLoaded', function () {
    initRevealAnimations();
    initLoadingStates();
    initTooltips();
    initFileUpload();
    initSearch();
    initSidebarToggle();
});

// Subtle scroll reveal (class-based, CSS handles the animation)
function initRevealAnimations() {
    if (!('IntersectionObserver' in window)) return;

    const observer = new IntersectionObserver((entries) => {
        entries.forEach((entry) => {
            if (entry.isIntersecting) {
                entry.target.classList.add('is-visible');
                observer.unobserve(entry.target);
            }
        });
    }, { threshold: 0.08, rootMargin: '0px 0px -40px 0px' });

    document.querySelectorAll('.reveal').forEach((el) => observer.observe(el));
}

function initLoadingStates() {
    // Add loading spinner on form submit without blocking submission
    const forms = document.querySelectorAll('form');
    forms.forEach((form) => {
        form.addEventListener('submit', function () {
            const submitBtn = form.querySelector('button[type="submit"]');
            if (submitBtn) {
                showLoading(submitBtn);
            }
        });
    });
}

function initTooltips() {
    if (typeof bootstrap === 'undefined') return;
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
}

function showLoading(button) {
    button.dataset.originalText = button.dataset.originalText || button.innerHTML;
    button.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Processando...';
    button.classList.add('loading');
}

// Mobile sidebar toggle
function initSidebarToggle() {
    const toggle = document.getElementById('sidebarToggle');
    const sidebar = document.querySelector('.mirach-sidebar');
    const backdrop = document.getElementById('sidebarBackdrop');
    if (!toggle || !sidebar) return;

    function close() {
        sidebar.classList.remove('is-open');
        if (backdrop) backdrop.classList.remove('is-visible');
    }

    toggle.addEventListener('click', function () {
        sidebar.classList.toggle('is-open');
        if (backdrop) backdrop.classList.toggle('is-visible');
    });
    if (backdrop) {
        backdrop.addEventListener('click', close);
    }
    sidebar.querySelectorAll('a').forEach((a) => a.addEventListener('click', close));
}

// File upload enhancements
function initFileUpload() {
    const fileInputs = document.querySelectorAll('input[type="file"]');

    fileInputs.forEach((input) => {
        input.addEventListener('change', function (e) {
            const file = e.target.files[0];
            if (file) {
                showFilePreview(file, this);
            }
        });

        input.addEventListener('dragover', function (e) {
            e.preventDefault();
            this.parentElement.classList.add('drag-over');
        });

        input.addEventListener('dragleave', function (e) {
            e.preventDefault();
            this.parentElement.classList.remove('drag-over');
        });

        input.addEventListener('drop', function (e) {
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
    const existing = input.parentElement.querySelector('.file-preview');
    if (existing) existing.remove();

    const preview = document.createElement('div');
    preview.className = 'file-preview';
    preview.innerHTML = `
        <div class="file-info">
            <i class="fas fa-file-excel text-success"></i>
            <span class="file-name">${file.name}</span>
            <span class="file-size">${formatFileSize(file.size)}</span>
        </div>
    `;
    input.parentElement.appendChild(preview);
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

    setTimeout(() => notification.classList.add('show'), 100);
    setTimeout(() => removeNotification(notification), duration);
    notification.querySelector('.notification-close').addEventListener('click', () => {
        removeNotification(notification);
    });
}

function removeNotification(notification) {
    notification.classList.remove('show');
    setTimeout(() => notification.remove(), 300);
}

function getNotificationIcon(type) {
    const icons = {
        success: 'check-circle',
        error: 'exclamation-circle',
        warning: 'exclamation-triangle',
        info: 'info-circle',
    };
    return icons[type] || 'info-circle';
}

// Search functionality
function initSearch() {
    const searchInputs = document.querySelectorAll('.search-input');

    searchInputs.forEach((input) => {
        input.addEventListener('input', function (e) {
            const query = e.target.value.toLowerCase();
            const items = document.querySelectorAll('.searchable-item');

            items.forEach((item) => {
                const text = item.textContent.toLowerCase();
                item.style.display = text.includes(query) ? '' : 'none';
            });
        });
    });
}
