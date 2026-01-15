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

const optionss_btn = document.querySelector(".options-btn");
const option = document.querySelector(".option");

optionss_btn.addEventListener("click", () => {
    var ds = option.style.display;
    if (ds === "none") {
        option.style.display = "";
    } else {
        option.style.display = "none";
    }
});

const add_files = document.querySelector(".add-files");
const files_cloud = document.querySelector("#fileInput");
const selectedFilesList = document.querySelector(".selectedFilesList");
let selectedFiles = [];

add_files.addEventListener("click", () => {
    files_cloud.click();
});

files_cloud.addEventListener("change", (event) => {
    const files = Array.from(event.target.files);
    selectedFiles.push(...files);
    files.forEach(file => {
        const li = document.createElement("li");
        li.className = "file";
        li.textContent = file.name;
        selectedFilesList.appendChild(li);
    });
    selectedFilesList.style.display = selectedFiles.length > 0 ? "" : "none";
});

const send_btn = document.querySelector(".send-btn");
const chat_wrapper = document.querySelector(".chat-wrapper");
const main_text = document.querySelector(".main-text");
const second_main_text = document.querySelector(".second-main-text");
const chat_input = document.querySelector('.input-chat');
const chat_text = document.querySelector(".chat-text");

send_btn.addEventListener("click", () => {
    const message = chat_input.value.trim();
    chat_input.value = "";
    main_text.style.display = "none";
    second_main_text.style.display = "none";
    chat_wrapper.classList.add('pinned-to-bottom');
    let user_text = document.createElement("p");
    user_text.className = "user-text";
    user_text.textContent = message;
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

        let model_text = document.createElement("p");
        model_text.className = "model-text";
        chat_text.appendChild(model_text);
        reader.read().then(function process({ done, value }) {
            if (done) {
                console.log("Поток завершён");
                return;
            }

            const chunk = decoder.decode(value, { stream: true });

            model_text.textContent += chunk
            return reader.read().then(process);
        });
    })
    .catch(err => {
        console.error("Ошибка потока:", err);
    });
});