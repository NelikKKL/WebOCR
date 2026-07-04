use image::imageops::FilterType;
use image::GenericImageView;
use tract_onnx::prelude::*;

/// Модель обучена на строках высотой 32px и переменной ширине.
/// Приводим любое входное изображение к этому формату, сохраняя пропорции,
/// чтобы одинаково хорошо работали разные размеры шрифта.
const TARGET_HEIGHT: u32 = 32;
const MIN_WIDTH: u32 = 16;

pub fn prepare_input(image_bytes: &[u8]) -> Result<Tensor, String> {
    let img = image::load_from_memory(image_bytes).map_err(|e| e.to_string())?;

    let (w, h) = img.dimensions();
    let aspect = w as f32 / h as f32;
    let target_w = ((TARGET_HEIGHT as f32) * aspect).round().max(MIN_WIDTH as f32) as u32;

    let resized = img
        .resize_exact(target_w, TARGET_HEIGHT, FilterType::Triangle)
        .to_luma8();

    // Нормализация в диапазон [-1, 1], как при обучении (см. train.py)
    let mut data = Vec::with_capacity((target_w * TARGET_HEIGHT) as usize);
    for pixel in resized.pixels() {
        let v = pixel[0] as f32 / 255.0;
        data.push((v - 0.5) / 0.5);
    }

    Tensor::from_shape(&[1, 1, TARGET_HEIGHT as usize, target_w as usize], &data)
        .map_err(|e| e.to_string())
}
