import init, { load_model, recognize_text } from "./webocr.js";

const resultEl = document.getElementById("result");
const previewEl = document.getElementById("preview");
const fileEl = document.getElementById("file");

async function main() {
  resultEl.textContent = "Загрузка WASM-модуля…";
  await init();

  resultEl.textContent = "Загрузка модели…";
  const modelResp = await fetch("model.onnx");
  const modelBytes = new Uint8Array(await modelResp.arrayBuffer());
  load_model(modelBytes);

  resultEl.textContent = "Готово. Загрузите изображение.";

  fileEl.addEventListener("change", async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    previewEl.src = URL.createObjectURL(file);
    previewEl.style.display = "block";

    resultEl.textContent = "Распознаю…";
    const bytes = new Uint8Array(await file.arrayBuffer());
    try {
      const text = recognize_text(bytes);
      resultEl.textContent = text || "(пусто)";
    } catch (err) {
      resultEl.textContent = "Ошибка: " + err;
    }
  });
}

main();
