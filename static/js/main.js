document.addEventListener('DOMContentLoaded', () => {
    let el = document.querySelector('.input-chat');
    const chat = el.closest('.chat');

    function createMeasureSpan() {
        const span = document.createElement('span');
        span.style.cssText = 'position:absolute;left:-9999px;white-space:pre;font:' + getComputedStyle(el).font;
        document.body.appendChild(span);
        return span;
    }

    let measureSpan = createMeasureSpan();

    function isOverflowing() {
        measureSpan.textContent = el.value || el.placeholder;
        return measureSpan.offsetWidth >= el.clientWidth - 4;
    }

    function switchToTextarea() {
        const textarea = document.createElement('textarea');
        textarea.className = 'input-chat';
        textarea.placeholder = 'Задай любой вопрос...';
        textarea.value = el.value;

        const styles = getComputedStyle(el);
        textarea.style.fontFamily = styles.fontFamily;
        textarea.style.fontSize = styles.fontSize;
        textarea.style.fontWeight = styles.fontWeight;
        textarea.style.fontStyle = styles.fontStyle;
        textarea.style.letterSpacing = styles.letterSpacing;
        textarea.style.textTransform = styles.textTransform;
        textarea.style.color = styles.color;
        textarea.style.backgroundColor = 'transparent';
        textarea.style.border = 'none';
        textarea.style.outline = 'none';
        textarea.style.resize = 'none';
        textarea.style.textAlign = styles.textAlign;
        textarea.style.padding = styles.padding;
        textarea.style.margin = styles.margin;
        textarea.style.lineHeight = styles.lineHeight;
        textarea.style.boxSizing = 'border-box';
        textarea.style.width = '100%';
        textarea.style.height = 'auto';
        textarea.style.overflow = 'hidden';

        el.replaceWith(textarea);
        el = textarea;
        textarea.focus();

        function updateHeight() {
            textarea.style.height = 'auto';
            const newHeight = Math.min(textarea.scrollHeight, 120);
            textarea.style.height = newHeight + 'px';
        }

        updateHeight();

        textarea.addEventListener('input', () => {
            updateHeight();
            if (textarea.value === '' || !isOverflowing()) {
                switchToInput();
            }
        });
    }

    function switchToInput() {
        const input = document.createElement('input');
        input.className = 'input-chat';
        input.placeholder = 'Задай любой вопрос...';
        input.value = el.value;

        el.replaceWith(input);
        el = input;
        input.focus();

        input.addEventListener('input', () => {
            if (isOverflowing()) {
                switchToTextarea();
            }
        });

        document.body.removeChild(measureSpan);
        measureSpan = createMeasureSpan();
    }

    el.addEventListener('input', () => {
        if (isOverflowing()) {
            switchToTextarea();
        }
    });
});

document.addEventListener("DOMContentLoaded", () => {
    const optionss_btn = document.querySelector(".options-btn");
    const option = document.querySelector(".option");
    const add_files = document.querySelector(".add-files");
    const files_cloud = document.querySelector("#fileInput");
    const selectedFilesList = document.querySelector(".selectedFilesList");
    const send_btn = document.querySelector(".send-btn");
    const chat_wrapper = document.querySelector(".chat-wrapper");
    const main_text = document.querySelector(".main-text");
    const second_main_text = document.querySelector(".second-main-text");
    const chat_input = document.querySelector('.input-chat');
    const chat_text = document.querySelector(".chat-text");

    let selectedFiles = [];

    optionss_btn.addEventListener("click", () => {
        option.style.display = option.style.display === "none" ? "block" : "none";
    });

    add_files.addEventListener("click", () => {
        files_cloud.click();
    });

    files_cloud.addEventListener("change", (event) => {
        const files = Array.from(event.target.files);
        selectedFiles.push(...files);
        selectedFilesList.innerHTML = "";
        files.forEach(file => {
            const li = document.createElement("li");
            li.textContent = file.name;
            selectedFilesList.appendChild(li);
        });
        selectedFilesList.style.display = "block";
    });

    send_btn.addEventListener("click", () => {
        const message = chat_input.value.trim();
        if (!message && selectedFiles.length === 0) return;

        chat_input.value = "";
        main_text.style.display = "none";
        second_main_text.style.display = "none";
        chat_wrapper.classList.add('pinned-to-bottom');

        const user_text = document.createElement("p");
        user_text.className = "user-text";
        user_text.textContent = message || "(файлы)";
        chat_text.appendChild(user_text);

        const formData = new FormData();
        formData.append("message", message);
        selectedFiles.forEach(file => {
            formData.append("files", file);
        });

        fetch("/chat", {
            method: "POST",
            body: formData
        })
        .then(response => {
            const reader = response.body.getReader();
            const decoder = new TextDecoder();

            // Очищаем файлы сразу после получения ответа
            selectedFiles = [];
            files_cloud.value = "";
            selectedFilesList.innerHTML = "";
            selectedFilesList.style.display = "none";

            const model_text = document.createElement("p");
            model_text.className = "model-text";
            chat_text.appendChild(model_text);

            return reader.read().then(function process({ done, value }) {
                if (done) return;
                const chunk = decoder.decode(value, { stream: true });
                model_text.textContent += chunk;
                chat_text.scrollTop = chat_text.scrollHeight;
                return reader.read().then(process);
            });
        })
        .catch(err => {
            console.error("Ошибка:", err);
            selectedFiles = [];
            files_cloud.value = "";
            selectedFilesList.innerHTML = "";
            selectedFilesList.style.display = "none";
        });
    });
});