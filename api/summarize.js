// ConsisAI Serverless API for Vercel
// Developed By Rajendra Singh Bisht

function semanticExtractiveSummary(text, style = 'bullets') {
  const rawSentences = text
    .replace(/([.?!])\s+(?=[A-Z0-9])/g, '$1|')
    .split('|')
    .map(s => s.trim())
    .filter(s => s.length > 15);

  if (rawSentences.length <= 1) {
    return text.trim();
  }

  const stopWords = new Set([
    'the','and','a','to','of','in','is','that','for','it','as','was','with','on','be','at','by','this',
    'have','from','or','one','had','by','word','but','not','what','all','were','we','when','your','can',
    'said','there','use','an','each','which','she','do','how','their','if','will','up','other','about',
    'out','many','then','them','these','so','some','her','would','make','like','him','into','time','has',
    'look','two','more','write','go','see','number','no','way','could','people','my','than','first','water',
    'been','call','who','oil','its','now','find','long','down','day','did','get','come','made','may','part'
  ]);

  const words = text.toLowerCase().match(/\b[a-z]{3,}\b/g) || [];
  const freq = {};
  for (const w of words) {
    if (!stopWords.has(w)) {
      freq[w] = (freq[w] || 0) + 1;
    }
  }

  const scored = rawSentences.map((sent, idx) => {
    const sWords = sent.toLowerCase().match(/\b[a-z]{3,}\b/g) || [];
    let score = 0;
    for (const w of sWords) {
      if (freq[w]) score += freq[w];
    }
    if (idx === 0) score *= 1.4;
    if (idx === rawSentences.length - 1) score *= 1.15;
    const lenPenalty = sent.length > 280 ? 0.8 : sent.length < 35 ? 0.7 : 1.0;
    return { sentence: sent, score: score * lenPenalty, index: idx };
  });

  if (style === 'oneliner') {
    scored.sort((a, b) => b.score - a.score);
    return scored[0].sentence.replace(/[.,;:]+$/, '') + '.';
  }

  const numSentences = style === 'bullets'
    ? Math.min(3, Math.max(2, Math.floor(rawSentences.length / 3)))
    : Math.min(4, Math.max(2, Math.floor(rawSentences.length / 2.5)));

  const topSentences = [...scored].sort((a, b) => b.score - a.score).slice(0, numSentences);
  topSentences.sort((a, b) => a.index - b.index);

  if (style === 'bullets') {
    return topSentences.map(item => `• ${item.sentence.replace(/[.,;:]+$/, '')}.`).join('\n\n');
  } else {
    return topSentences.map(item => item.sentence.replace(/[.,;:]+$/, '') + '.').join(' ');
  }
}

export default async function handler(req, res) {
  // CORS Headers
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
      return res.status(400).json({ error: 'Please enter text to summarize.' });
    }

    const token = (customToken && customToken.trim()) || process.env.HF_TOKEN || '';
    let summary = '';
    let mode = 'Fast Semantic Engine';

    // If token is provided, attempt Hugging Face Router
    if (token) {
      try {
        const isQwen = model.toLowerCase().includes('qwen') || model.toLowerCase().includes('instruct');
        let payload = {};

        if (isQwen) {
          let instruction = 'Provide a high-density executive brief with 2-4 key takeaway bullet points.';
          if (style === 'oneliner') {
            instruction = 'Provide a single concise punchline sentence summarizing the core takeaway.';
          } else if (style === 'paragraph') {
            instruction = 'Provide a single fluent synthesis paragraph preserving the exact core meaning.';
          }

          payload = {
            inputs: `<|im_start|>system\nYou are ConsisAI, an executive summarizer developed by Rajendra Singh Bisht. ${instruction}<|im_end|>\n<|im_start|>user\n${text}<|im_end|>\n<|im_start|>assistant\n`,
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

        const hfUrl = `https://router.huggingface.co/hf-inference/models/${model}`;
        const hfRes = await fetch(hfUrl, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${token}`
          },
          body: JSON.stringify(payload)
        });

        if (hfRes.ok) {
          const hfData = await hfRes.json();
          if (Array.isArray(hfData) && hfData[0]) {
            summary = hfData[0].summary_text || hfData[0].generated_text || '';
          } else if (hfData.generated_text) {
            summary = hfData.generated_text;
          }
          if (summary) {
            mode = `Deep Neural (${model.split('/').pop()})`;
          }
        } else {
          console.warn(`HF Router returned ${hfRes.status}, using semantic fallback`);
        }
      } catch (hfErr) {
        console.warn('HF Router connection issue, falling back to semantic engine:', hfErr.message);
      }
    }

    // Fallback if no token or if HF endpoint is busy/unreachable
    if (!summary) {
      summary = semanticExtractiveSummary(text, style);
      mode = token ? 'ConsisAI Semantic Engine (HF Busy)' : 'ConsisAI Instant Cloud Semantic';
    }

    summary = summary.trim();

    // Word analytics
    const origWords = text.trim().split(/\s+/).filter(Boolean).length;
    const summWords = summary.trim().split(/\s+/).filter(Boolean).length;
    const ratio = origWords > 0
      ? Math.max(0, ((origWords - summWords) / origWords) * 100).toFixed(1)
      : '0.0';

    return res.status(200).json({
      summary,
      original_word_count: origWords,
      summary_word_count: summWords,
      compression_ratio: parseFloat(ratio),
      engine_mode: mode
    });
  } catch (err) {
    console.error('API Handler Error:', err);
    return res.status(500).json({ error: err.message || 'Internal server error' });
  }
}
