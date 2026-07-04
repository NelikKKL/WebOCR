use once_cell::sync::OnceCell;
use std::sync::Mutex;
use tract_onnx::prelude::*;
use wasm_bindgen::prelude::*;

mod ctc_decode;
mod preprocess;

// Набор символов должен строго совпадать с тем, что использовался при обучении
// (см. training/generate_dataset.py). Индекс 0 зарезервирован под CTC-blank.
const CHARSET: &str =
    " !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~";

type Model = SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>;

static MODEL: OnceCell<Mutex<Model>> = OnceCell::new();

#[wasm_bindgen(start)]
pub fn init_panic_hook() {
    console_error_panic_hook::set_once();
}

/// Загружает ONNX-модель (байты файла model.onnx) в память один раз при старте страницы.
#[wasm_bindgen]
pub fn load_model(model_bytes: &[u8]) -> Result<(), JsValue> {
    let model = tract_onnx::onnx()
        .model_for_read(&mut std::io::Cursor::new(model_bytes))
        .map_err(|e| JsValue::from_str(&format!("ошибка чтения модели: {e}")))?
        .into_optimized()
        .map_err(|e| JsValue::from_str(&format!("ошибка оптимизации: {e}")))?
        .into_runnable()
        .map_err(|e| JsValue::from_str(&format!("ошибка компиляции графа: {e}")))?;

    MODEL
        .set(Mutex::new(model))
        .map_err(|_| JsValue::from_str("модель уже загружена"))?;
    Ok(())
}

/// Принимает сырые байты изображения (png/jpeg), возвращает распознанный текст.
#[wasm_bindgen]
pub fn recognize_text(image_bytes: &[u8]) -> Result<String, JsValue> {
    let model = MODEL
        .get()
        .ok_or_else(|| JsValue::from_str("модель не загружена: вызовите load_model()"))?;
    let model = model
        .lock()
        .map_err(|_| JsValue::from_str("не удалось получить блокировку модели"))?;

    let input = preprocess::prepare_input(image_bytes).map_err(JsValue::from_str)?;

    let outputs = model
        .run(tvec!(input.into()))
        .map_err(|e| JsValue::from_str(&format!("ошибка инференса: {e}")))?;

    // Выход модели имеет форму (T, 1, num_classes) — см. model.py, permute(1,0,2)
    let output = outputs[0]
        .to_array_view::<f32>()
        .map_err(|e| JsValue::from_str(&format!("ошибка чтения выхода: {e}")))?;

    Ok(ctc_decode::greedy_decode(output, CHARSET))
}
