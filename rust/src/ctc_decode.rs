use ndarray::ArrayViewD;

/// Простое жадное (greedy) CTC-декодирование: на каждом временном шаге берём
/// самый вероятный класс, схлопываем повторы и убираем blank (индекс 0).
/// Для более высокой точности можно заменить на beam search, но greedy
/// значительно проще и быстрее — обычно этого достаточно для коротких строк.
pub fn greedy_decode(output: ArrayViewD<f32>, charset: &str) -> String {
    let chars: Vec<char> = charset.chars().collect();
    const BLANK_IDX: usize = 0;

    let shape = output.shape();
    // Ожидаем форму (time_steps, batch=1, num_classes)
    let (time_steps, num_classes) = (shape[0], shape[shape.len() - 1]);

    let mut result = String::new();
    let mut prev = usize::MAX;

    for t in 0..time_steps {
        let mut best_idx = 0usize;
        let mut best_val = f32::MIN;
        for c in 0..num_classes {
            let v = output[[t, 0, c]];
            if v > best_val {
                best_val = v;
                best_idx = c;
            }
        }

        if best_idx != BLANK_IDX && best_idx != prev {
            if let Some(&ch) = chars.get(best_idx - 1) {
                result.push(ch);
            }
        }
        prev = best_idx;
    }

    result
}
