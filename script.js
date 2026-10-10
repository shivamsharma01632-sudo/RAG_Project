// Local backend while developing, Render backend once deployed. Update the Render URL if yours differs.
const API_URL = ["localhost", "127.0.0.1"].includes(location.hostname)
  ? "http://127.0.0.1:8000"
  : "https://folio-rag-api.onrender.com";
const pdfInput = document.querySelector("#pdf-input");
const uploadZone = document.querySelector("#upload-zone");
const replaceButton = document.querySelector("#replace-button");
const documentStatus = document.querySelector("#document-status");
const documentName = document.querySelector("#document-name");
const questionForm = document.querySelector("#question-form");
const questionInput = document.querySelector("#question-input");
const sendButton = document.querySelector("#send-button");
const conversation = document.querySelector("#conversation");
const emptyState = document.querySelector("#empty-state");

let documentReady = false;
let busy = false;

pdfInput.addEventListener("change", () => {
  if (pdfInput.files[0]) uploadPdf(pdfInput.files[0]);
});
replaceButton.addEventListener("click", () => pdfInput.click());

for (const eventName of ["dragenter", "dragover"]) {
  uploadZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    uploadZone.classList.add("is-dragging");
  });
}
for (const eventName of ["dragleave", "drop"]) {
  uploadZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    uploadZone.classList.remove("is-dragging");
  });
}
uploadZone.addEventListener("drop", (event) => {
  const [file] = event.dataTransfer.files;
  if (file) uploadPdf(file);
});

async function uploadPdf(file) {
  if (!file.name.toLowerCase().endsWith(".pdf")) {
    setStatus("Please choose a PDF file.", "error");
    return;
  }
  if (file.size > 25 * 1024 * 1024) {
    setStatus("This PDF is larger than 25 MB.", "error");
    return;
  }

  setBusy(true);
  setStatus("Checking document index...", "loading");
  uploadZone.classList.add("is-loading");
  const body = new FormData();
  body.append("file", file);

  try {
    const response = await fetch(`${API_URL}/upload`, { method: "POST", body });
    const result = await response.json();
    if (!response.ok) throw new Error(result.detail || "Could not process this PDF.");
    documentReady = true;
    documentName.textContent = result.filename;
    documentStatus.classList.remove("is-error", "is-loading");
    documentStatus.classList.add("is-ready");
    replaceButton.hidden = false;
    questionInput.disabled = false;
    questionInput.placeholder = "Ask a question about your document...";
    questionInput.focus();
    const indexingMessage = result.index_status === "reused"
      ? `${result.filename} — existing embeddings loaded (${result.chunk_count} chunks)`
      : `${result.filename} — indexed ${result.chunk_count} chunks`;
    setStatus(indexingMessage, "ready");
    resetConversation();
    emptyState.querySelector("p").textContent = result.index_status === "reused"
      ? "Your PDF was already indexed, so its existing embeddings are ready to use."
      : "Your PDF is indexed and ready. Ask a question and Folio will look for the answer in its pages.";
  } catch (error) {
    documentReady = false;
    setStatus(error.message || "Unable to reach the API. Is the backend running?", "error");
  } finally {
    uploadZone.classList.remove("is-loading");
    setBusy(false);
    pdfInput.value = "";
  }
}

questionForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const question = questionInput.value.trim();
  if (!question || !documentReady || busy) return;
  addMessage(question, "user");
  questionInput.value = "";
  questionInput.style.height = "auto";
  const pendingMessage = addMessage("Searching your document...", "assistant", true);
  setBusy(true);

  try {
    const response = await fetch(`${API_URL}/path`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.detail || "The question could not be answered.");
    pendingMessage.textContent = result.answer;
    pendingMessage.classList.remove("is-pending");
  } catch (error) {
    pendingMessage.textContent = error.message || "Unable to reach the API. Please try again.";
    pendingMessage.classList.remove("is-pending");
    pendingMessage.classList.add("is-error");
  } finally {
    setBusy(false);
    questionInput.focus();
  }
});

questionInput.addEventListener("input", () => {
  questionInput.style.height = "auto";
  questionInput.style.height = `${Math.min(questionInput.scrollHeight, 140)}px`;
});

function addMessage(text, role, pending = false) {
  emptyState.hidden = true;
  const message = document.createElement("article");
  message.className = `message message-${role}${pending ? " is-pending" : ""}`;
  const label = document.createElement("span");
  label.className = "message-label";
  label.textContent = role === "user" ? "YOU" : "FOLIO";
  const content = document.createElement("p");
  content.textContent = text;
  message.append(label, content);
  conversation.append(message);
  conversation.scrollTop = conversation.scrollHeight;
  return content;
}

function resetConversation() {
  conversation.replaceChildren(emptyState);
  emptyState.hidden = false;
}

function setStatus(text, state) {
  documentName.textContent = text;
  documentStatus.classList.toggle("is-ready", state === "ready");
  documentStatus.classList.toggle("is-loading", state === "loading");
  documentStatus.classList.toggle("is-error", state === "error");
}

function setBusy(value) {
  busy = value;
  sendButton.disabled = value || !documentReady;
  questionInput.disabled = value || !documentReady;
  sendButton.classList.toggle("is-busy", value);
}
