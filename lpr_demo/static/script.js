const dropzone = document.getElementById("dropzone");
const dropzoneInner = document.getElementById("dropzoneInner");
const fileInput = document.getElementById("fileInput");
const previewImage = document.getElementById("previewImage");
const scanSweep = document.getElementById("scanSweep");

const warningBanner = document.getElementById("warningBanner");
const pipelineSection = document.getElementById("pipeline");
const pipelineSteps = document.getElementById("pipelineSteps");
const resultSection = document.getElementById("resultSection");
const plateReadout = document.getElementById("plateReadout");
const charStrip = document.getElementById("charStrip");
const resetBtn = document.getElementById("resetBtn");

const statusDot = document.getElementById("statusDot");
const statusText = document.getElementById("statusText");

const STEP_DELAY_MS = 260;

init();

function init() {
  checkModelHealth();

  dropzone.addEventListener("click", () => fileInput.click());
  dropzone.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      fileInput.click();
    }
  });
  dropzone.tabIndex = 0;
  dropzone.setAttribute("role", "button");
  dropzone.setAttribute("aria-label", "Chọn ảnh biển số để tải lên");

  fileInput.addEventListener("change", (e) => {
    const file = e.target.files[0];
    if (file) handleFile(file);
  });

  ["dragenter", "dragover"].forEach((evt) =>
    dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropzone.classList.add("dragover");
    })
  );

  ["dragleave", "drop"].forEach((evt) =>
    dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropzone.classList.remove("dragover");
    })
  );

  dropzone.addEventListener("drop", (e) => {
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  });

  resetBtn.addEventListener("click", resetDemo);
}

async function checkModelHealth() {
  try {
    const res = await fetch("/api/health");
    const data = await res.json();
    if (data.model_loaded) {
      statusDot.classList.add("ok");
      statusText.textContent = "Model SVM đã sẵn sàng";
    } else {
      statusDot.classList.add("warn");
      statusText.textContent = "Chưa nạp được model SVM";
    }
  } catch (err) {
    statusText.textContent = "Không kết nối được backend";
  }
}

function handleFile(file) {
  if (!file.type.startsWith("image/")) return;

  const url = URL.createObjectURL(file);
  previewImage.src = url;
  previewImage.hidden = false;
  dropzoneInner.hidden = true;
  scanSweep.hidden = false;

  resultSection.hidden = true;
  warningBanner.hidden = true;
  pipelineSection.hidden = true;
  pipelineSteps.innerHTML = "";

  uploadAndProcess(file);
}

async function uploadAndProcess(file) {
  const formData = new FormData();
  formData.append("file", file);

  try {
    const res = await fetch("/api/process", {
      method: "POST",
      body: formData,
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.error || "Xử lý ảnh thất bại");
    }

    const data = await res.json();
    scanSweep.hidden = true;
    renderResults(data);
  } catch (err) {
    scanSweep.hidden = true;
    showWarning(err.message || "Đã có lỗi xảy ra, thử lại ảnh khác nhé.");
  }
}

function renderResults(data) {
  if (data.model_warning) {
    showWarning(data.model_warning);
  }

  pipelineSection.hidden = false;
  pipelineSteps.innerHTML = "";

  data.steps.forEach((step, index) => {
    const card = document.createElement("div");
    card.className = "step-card";
    card.style.animationDelay = `${index * STEP_DELAY_MS}ms`;

    card.innerHTML = `
      <div class="step-card__media">
        <img src="${step.image}" alt="${escapeHtml(step.title)}" />
      </div>
      <div>
        <p class="step-card__title">${escapeHtml(step.title)}</p>
        <p class="step-card__desc">${escapeHtml(step.description)}</p>
      </div>
    `;
    pipelineSteps.appendChild(card);
  });

  const resultDelay = data.steps.length * STEP_DELAY_MS + 200;
  setTimeout(() => renderFinalResult(data), resultDelay);
}

function renderFinalResult(data) {
  resultSection.hidden = false;
  resultSection.style.animationDelay = "0ms";

  plateReadout.innerHTML = "";
  charStrip.innerHTML = "";

  if (!data.predicted_text) {
    const empty = document.createElement("p");
    empty.className = "step-card__desc";
    empty.textContent = "Không phát hiện được ký tự nào trong ảnh này.";
    plateReadout.appendChild(empty);
    return;
  }

  for (const ch of data.predicted_text) {
    const tile = document.createElement("div");
    tile.className = "plate-readout__char";
    tile.textContent = ch;
    plateReadout.appendChild(tile);
  }

  data.characters.forEach((charData, index) => {
    const tile = document.createElement("div");
    tile.className = "char-tile";
    tile.innerHTML = `
      <img src="${charData.processed}" alt="Ký tự ${index + 1}" />
      <div class="char-tile__label">${escapeHtml(charData.prediction)}</div>
    `;
    charStrip.appendChild(tile);
  });

  resultSection.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function showWarning(message) {
  warningBanner.hidden = false;
  warningBanner.textContent = message;
}

function resetDemo() {
  previewImage.src = "";
  previewImage.hidden = true;
  dropzoneInner.hidden = false;
  fileInput.value = "";

  pipelineSection.hidden = true;
  resultSection.hidden = true;
  warningBanner.hidden = true;
  pipelineSteps.innerHTML = "";
  plateReadout.innerHTML = "";
  charStrip.innerHTML = "";
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}
