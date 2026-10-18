// ConsisAI Serverless API for Vercel
// Developed By Rajendra Singh Bisht

export default async function handler(req, res) {
  // CORS support
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization');

  if (req.method === 'OPTIONS') {
    return res.status(200).end();
  }

  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed. Use POST.' });
  }

  try {
    const {
      text,
      model = 'facebook/bart-large-cnn',
      style = 'bullets',
      maxLength = 150,
      customToken = ''
    } = req.body || {};

    if (!text || !text.trim()) {
      return res.status(400).json({ error: 'Please provide text to summarize.' });
    }

    const token = customToken || process.env.HF_TOKEN || '';
    const headers = { 'Content-Type': 'application/json' };
    if (token) {
      headers['Authorization'] = `Bearer ${token}`;
    }

    let payload = {};
    const isQwen = model.toLowerCase().includes('qwen') || model.toLowerCase().includes('instruct');

    if (isQwen) {
      let instruction = "Provide a high-density executive summary with 2-4 bullet points.";
      if (style === 'oneliner') {
        instruction = "Provide a single concise punchline sentence summarizing the core takeaway.";
      } else if (style === 'paragraph') {
        instruction = "Provide a single fluent synthesis paragraph preserving the exact core meaning.";
      }

      const prompt = `<|im_start|>system\nYou are ConsisAI, an executive summarization engine developed by Rajendra Singh Bisht. ${instruction}<|im_end|>\n<|im_start|>user\n${text}<|im_end|>\n<|im_start|>assistant\n`;
      payload = {
        inputs: prompt,
        parameters: {
          max_new_tokens: parseInt(maxLength, 10) || 150,
          temperature: 0.3,
          return_full_text: false
        }
      };
    } else {
      payload = {
        inputs: text,
        parameters: {
          max_length: parseInt(maxLength, 10) || 150,
          min_length: Math.max(20, Math.floor((parseInt(maxLength, 10) || 150) / 4)),
          do_sample: false
        }
      };
    }

    const hfUrl = `https://api-inference.huggingface.co/models/${model}`;
    const response = await fetch(hfUrl, {
      method: 'POST',
      headers,
      body: JSON.stringify(payload)
    });

    const data = await response.json();

    if (!response.ok) {
      const errMsg = data.error || (data.estimated_time ? `Model is warming up (${Math.round(data.estimated_time)}s). Please retry in a moment.` : 'Hugging Face API request failed.');
      return res.status(response.status).json({ error: errMsg });
    }

    let summary = '';
    if (Array.isArray(data) && data[0]) {
      summary = data[0].summary_text || data[0].generated_text || JSON.stringify(data[0]);
    } else if (data.generated_text) {
      summary = data.generated_text;
    } else if (typeof data === 'string') {
      summary = data;
    } else {
      summary = JSON.stringify(data);
    }

    summary = summary.trim();

    // Compute analytics
    const origWords = text.trim().split(/\s+/).filter(Boolean).length;
    const summWords = summary.trim().split(/\s+/).filter(Boolean).length;
    const ratio = origWords > 0 ? Math.max(0, ((origWords - summWords) / origWords) * 100).toFixed(1) : '0.0';

    return res.status(200).json({
      summary,
      original_word_count: origWords,
      summary_word_count: summWords,
      compression_ratio: parseFloat(ratio),
      model_used: model
    });
  } catch (err) {
    console.error('Summarization Error:', err);
    return res.status(500).json({ error: err.message || 'Internal server error' });
  }
}
