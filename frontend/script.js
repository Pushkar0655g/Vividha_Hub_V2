const processModeInputs = document.querySelectorAll('input[name="processMode"]');
const { ipcRenderer } = require("electron");

const videoPathInput = document.getElementById("videoPath");
const outputDirInput = document.getElementById("outputDir");
const inputLang = document.getElementById("inputLang");
const audioLang = document.getElementById("audioLang");
const subtitleLang = document.getElementById("subtitleLang");
const browseVideoBtn = document.getElementById("browseVideoBtn");
const browseOutputBtn = document.getElementById("browseOutputBtn");
const processBtn = document.getElementById("processBtn");
const progressBar = document.getElementById("progressBar");
const progressText = document.getElementById("progressText");
const logs = document.getElementById("logs");

browseVideoBtn.addEventListener("click", async () => {
  const selected = await ipcRenderer.invoke("pick-video");
  if (selected) {
    videoPathInput.value = selected;
  }
});

browseOutputBtn.addEventListener("click", async () => {
  const selected = await ipcRenderer.invoke("pick-output-folder");
  if (selected) {
    outputDirInput.value = selected;
  }
});

processBtn.addEventListener("click", async () => {
  const videoPath = videoPathInput.value.trim();
  const outputDir = outputDirInput.value.trim();

  if (!videoPath) {
    alert("Please choose a video file.");
    return;
  }

  if (!outputDir) {
    alert("Please choose an output folder.");
    return;
  }

  progressBar.style.width = "0%";
  progressText.textContent = "Starting...";
  logs.textContent = "";
  processBtn.disabled = true;
  processBtn.textContent = "Processing...";

  let selectedMode = "subtitles";
  for (const input of processModeInputs) {
    if (input.checked) {
      selectedMode = input.value;
      break;
    }
  }

  const enableDubbing = selectedMode === "dubbing";

  const result = await ipcRenderer.invoke("start-processing", {
    videoPath,
    outputDir,
    inputLang: inputLang.value,
    audioLang: audioLang.value,
    subtitleLang: subtitleLang.value,
    enableDubbing
  });

  processBtn.disabled = false;
  processBtn.textContent = "Process Video";

  if (result.status === "complete") {
    progressBar.style.width = "100%";
    progressText.textContent = "Completed successfully";
    logs.textContent += `\nDone.\nOutput: ${result.output}\n`;
    alert(`Processing completed successfully.\n\nOutput:\n${result.output}`);
  } else {
    progressText.textContent = "Failed";
    logs.textContent += `\nError: ${result.message}\n`;
    alert(`Processing failed: ${result.message}`);
  }
});

ipcRenderer.on("progress-update", (event, data) => {
  if (typeof data.percent === "number") {
    const safePercent = Math.max(0, Math.min(100, data.percent));
    progressBar.style.width = `${safePercent}%`;
  }

  if (data.message) {
    progressText.textContent = data.message;
  }
});

ipcRenderer.on("process-log", (event, message) => {
  logs.textContent += message;
  logs.scrollTop = logs.scrollHeight;
});
function updateModeUI() {
  let selectedMode = "subtitles";

  for (const input of processModeInputs) {
    if (input.checked) {
      selectedMode = input.value;
      break;
    }
  }

  const dubbingEnabled = selectedMode === "dubbing";
  audioLang.disabled = !dubbingEnabled;
}

for (const input of processModeInputs) {
  input.addEventListener("change", updateModeUI);
}

updateModeUI();