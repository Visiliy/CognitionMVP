async function fetchUserFiles() {
    try {
        const response = await fetch('/user_files');
        if (response.ok) {
            const files = await response.json();
            const userFilesList = document.getElementById('userFilesList');
            userFilesList.innerHTML = files.length === 0 ? '<li style="color: #888; cursor: default;">Нет загруженных файлов</li>' : '';
            files.forEach(file => {
                const li = document.createElement('li');
                li.textContent = file;
                userFilesList.appendChild(li);
            });
        }
    } catch (e) {
        console.error('Ошибка загрузки файлов');
    }
}

function displayMessage(message, sender) {
    const chatText = document.getElementById('chatText');
    const chatMainWrapper = document.getElementById('chatMainWrapper');
    const messageDiv = document.createElement('div');
    messageDiv.classList.add('chat-message', sender);
    const bubble = document.createElement('div');
    bubble.classList.add('bubble');
    bubble.textContent = message;
    messageDiv.appendChild(bubble);
    chatText.appendChild(messageDiv);
    chatText.scrollTop = chatText.scrollHeight;
    chatMainWrapper.classList.add('has-messages');
}

function adjustTextareaHeight(textarea) {
    textarea.style.height = 'auto';
    textarea.style.height = Math.min(textarea.scrollHeight, 140) + 'px';
}

function closeSidebar() {
    const sidebar = document.getElementById('sidebar');
    const overlay = document.querySelector('.sidebar-overlay');
    sidebar.classList.remove('open');
    overlay.classList.remove('open');
}

document.addEventListener('DOMContentLoaded', () => {
    const elements = {
        loginRegisterBtn: document.getElementById('loginRegisterBtn'),
        logoutBtn: document.getElementById('logoutBtn'),
        authModal: document.getElementById('authModal'),
        closeButton: document.querySelector('.close-button'),
        loginForm: document.getElementById('loginForm'),
        registerForm: document.getElementById('registerForm'),
        showRegister: document.getElementById('showRegister'),
        showLogin: document.getElementById('showLogin'),
        loginSubmit: document.getElementById('loginSubmit'),
        registerSubmit: document.getElementById('registerSubmit'),
        sidebarToggle: document.getElementById('sidebarToggle'),
        sidebar: document.getElementById('sidebar'),
        optionsBtn: document.getElementById('optionsBtn'),
        sendBtn: document.getElementById('sendBtn'),
        inputChat: document.getElementById('inputChat'),
        chatText: document.getElementById('chatText'),
        selectedFilesList: document.getElementById('selectedFilesList'),
        sidebarFileInput: document.getElementById('sidebarFileInput'),
        sidebarUploadBtn: document.getElementById('sidebarUploadBtn'),
        chatMainWrapper: document.getElementById('chatMainWrapper')
    };

    let selectedFiles = [];
    const fileInput = document.createElement('input');
    fileInput.type = 'file';
    fileInput.multiple = true;
    fileInput.accept = '.pdf,.txt,.docx';
    fileInput.style.display = 'none';
    document.body.appendChild(fileInput);

    if (elements.sidebarToggle) {
        elements.sidebarToggle.addEventListener('click', () => {
            elements.sidebar.classList.add('open');
            const overlay = document.createElement('div');
            overlay.className = 'sidebar-overlay';
            overlay.addEventListener('click', closeSidebar);
            document.body.appendChild(overlay);
            overlay.classList.add('open');
        });
    }

    if (elements.loginRegisterBtn) {
        elements.loginRegisterBtn.addEventListener('click', () => {
            elements.authModal.style.display = 'flex';
            elements.loginForm.style.display = 'flex';
            elements.registerForm.style.display = 'none';
        });
    }

    if (elements.closeButton) {
        elements.closeButton.addEventListener('click', () => {
            elements.authModal.style.display = 'none';
        });
    }

    window.addEventListener('click', (e) => {
        if (e.target === elements.authModal) {
            elements.authModal.style.display = 'none';
        }
    });

    if (elements.showRegister) {
        elements.showRegister.addEventListener('click', (e) => {
            e.preventDefault();
            elements.loginForm.style.display = 'none';
            elements.registerForm.style.display = 'flex';
        });
    }

    if (elements.showLogin) {
        elements.showLogin.addEventListener('click', (e) => {
            e.preventDefault();
            elements.registerForm.style.display = 'none';
            elements.loginForm.style.display = 'flex';
        });
    }

    if (elements.registerSubmit) {
        elements.registerSubmit.addEventListener('click', async (e) => {
            e.preventDefault();
            const username = document.getElementById('registerUsername').value.trim();
            const password = document.getElementById('registerPassword').value;
            
            if (!username || !password) {
                alert('Заполните все поля');
                return;
            }

            const formData = new FormData();
            formData.append('username', username);
            formData.append('password', password);

            try {
                const response = await fetch('/register', {
                    method: 'POST',
                    body: formData
                });

                if (response.ok) {
                    elements.authModal.style.display = 'none';
                    window.location.reload();
                } else {
                    const errorText = await response.text();
                    alert('Ошибка регистрации: ' + (errorText || 'Неизвестная ошибка'));
                }
            } catch (error) {
                alert('Ошибка сети');
            }
        });
    }

    if (elements.loginSubmit) {
        elements.loginSubmit.addEventListener('click', async (e) => {
            e.preventDefault();
            const username = document.getElementById('loginUsername').value.trim();
            const password = document.getElementById('loginPassword').value;
            
            if (!username || !password) {
                alert('Заполните все поля');
                return;
            }

            const formData = new FormData();
            formData.append('username', username);
            formData.append('password', password);

            try {
                const response = await fetch('/login', {
                    method: 'POST',
                    body: formData
                });

                if (response.ok) {
                    elements.authModal.style.display = 'none';
                    window.location.reload();
                } else {
                    const errorText = await response.text();
                    alert('Ошибка входа: ' + (errorText || 'Неверные данные'));
                }
            } catch (error) {
                alert('Ошибка сети');
            }
        });
    }

    if (elements.logoutBtn) {
        elements.logoutBtn.addEventListener('click', async () => {
            try {
                await fetch('/logout');
                window.location.reload();
            } catch (e) {}
        });
    }

    if (document.querySelector('.username')) {
        fetchUserFiles();
    }

    elements.inputChat.addEventListener('input', () => adjustTextareaHeight(elements.inputChat));

    if (elements.optionsBtn) {
        elements.optionsBtn.addEventListener('click', () => fileInput.click());
    }

    fileInput.addEventListener('change', (e) => {
        selectedFiles = Array.from(e.target.files);
        updateSelectedFilesList();
    });

    function updateSelectedFilesList() {
        elements.selectedFilesList.innerHTML = '';
        if (selectedFiles.length > 0) {
            elements.selectedFilesList.style.display = 'flex';
            selectedFiles.forEach((file, index) => {
                const li = document.createElement('li');
                li.textContent = file.name;
                const removeBtn = document.createElement('span');
                removeBtn.className = 'remove-file';
                removeBtn.textContent = '×';
                removeBtn.addEventListener('click', () => {
                    selectedFiles.splice(index, 1);
                    updateSelectedFilesList();
                });
                li.appendChild(removeBtn);
                elements.selectedFilesList.appendChild(li);
            });
        } else {
            elements.selectedFilesList.style.display = 'none';
        }
    }

    if (elements.sendBtn) {
        elements.sendBtn.addEventListener('click', async () => {
            if (!document.querySelector('.username')) {
                alert('Требуется авторизация');
                return;
            }

            const message = elements.inputChat.value.trim();
            if (!message && selectedFiles.length === 0) return;

            displayMessage(message || '[файлы]', 'user');

            elements.inputChat.value = '';
            elements.selectedFilesList.innerHTML = '';
            selectedFiles = [];
            adjustTextareaHeight(elements.inputChat);

            const formData = new FormData();
            formData.append('message', message);
            selectedFiles.forEach(file => formData.append('files', file));

            try {
                const response = await fetch('/chat', { method: 'POST', body: formData });
                const reader = response.body.getReader();
                const decoder = new TextDecoder('utf-8');
                let aiResponse = '';

                const aiMessageDiv = document.createElement('div');
                aiMessageDiv.classList.add('chat-message', 'ai');
                const aiBubble = document.createElement('div');
                aiBubble.className = 'bubble';
                aiMessageDiv.appendChild(aiBubble);
                elements.chatText.appendChild(aiMessageDiv);

                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;
                    aiResponse += decoder.decode(value, { stream: true });
                    aiBubble.textContent = aiResponse;
                    elements.chatText.scrollTop = elements.chatText.scrollHeight;
                }
            } catch (error) {
                displayMessage('Произошла ошибка', 'ai');
            }
        });
    }

    elements.inputChat.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            elements.sendBtn.click();
        }
    });

    if (elements.sidebarUploadBtn) {
        elements.sidebarUploadBtn.addEventListener('click', () => elements.sidebarFileInput.click());
    }

    if (elements.sidebarFileInput) {
        elements.sidebarFileInput.addEventListener('change', async (e) => {
            const files = Array.from(e.target.files);
            if (!files.length) return;

            const formData = new FormData();
            files.forEach(file => formData.append('files', file));

            try {
                const response = await fetch('/upload', { method: 'POST', body: formData });
                if (response.ok) {
                    alert('Файлы загружены');
                    fetchUserFiles();
                    elements.sidebarFileInput.value = '';
                    closeSidebar();
                } else {
                    alert('Ошибка загрузки');
                }
            } catch (error) {
                alert('Ошибка загрузки');
            }
        });
    }
});
