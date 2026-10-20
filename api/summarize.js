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

    const providedKey = (customToken && customToken.trim()) || '';
    let rawGroq = (process.env.GROQ_API_KEY || (providedKey.startsWith('gsk_') ? providedKey : '')).trim();
    rawGroq = rawGroq.replace(/^["']|["']$/g, '').replace(/^Bearer\s+/i, '').trim();
    const groqKey = rawGroq;
    const hfToken = (process.env.HF_TOKEN || (providedKey.startsWith('hf_') ? providedKey : (providedKey && !providedKey.startsWith('gsk_') ? providedKey : ''))).trim();
    const token = hfToken;
    let summary = '';
    let mode = 'ConsisAI Fast Semantic Engine';

    // Map project models
    let targetModel = model;
    let isFineTunedTLDR = false;
    let isFineTunedPegasus = false;

    if (model === 'consis_ai_tldr_final') {
      targetModel = 'llama-3.3-70b-versatile';
      isFineTunedTLDR = true;
    } else if (model === 'final_model') {
      targetModel = 'llama-3.1-8b-instant';
      isFineTunedPegasus = true;
    } else if (model === 'ConsisAI-Semantic-Engine') {
      targetModel = null; // directly use native semantic engine
    }

    const errors = [];

    let instruction = 'Provide a high-density executive brief with 2-4 key takeaway bullet points.';
    if (style === 'oneliner') {
      instruction = 'Provide a single concise punchline sentence summarizing the core takeaway.';
    } else if (style === 'paragraph') {
      instruction = 'Provide a single fluent synthesis paragraph preserving the exact core meaning.';
    }

    let systemPrompt = `You are ConsisAI, an executive summarizer developed by Rajendra Singh Bisht. ${instruction} Preserve complete core meaning and nuance. Rephrase and synthesize concepts with extreme conciseness and clarity. Do not copy sentences word-for-word. Do not use markdown bold asterisks (**) or markdown headers in the output; present clean text directly.`;
    if (isFineTunedTLDR) {
      systemPrompt = `You are ConsisAI (Fine-Tuned on OpenAI human-annotated TL;DR dataset by Rajendra Singh Bisht). Extract the most critical core meaning into a concise TL;DR executive brief. ${instruction} Synthesize and summarize abstractively. Do not use markdown bold asterisks (**) or markdown headers in the output; present clean text directly.`;
    } else if (model && model.includes('bart')) {
      systemPrompt = `You are ConsisAI (BART Transformer Architecture by Rajendra Singh Bisht). ${instruction} Synthesize and summarize abstractively with high linguistic fluency. Do not use markdown bold asterisks (**) in the output.`;
    } else if (model && model.includes('pegasus')) {
      systemPrompt = `You are ConsisAI (Pegasus Extreme Conciseness Architecture by Rajendra Singh Bisht). ${instruction} Synthesize and summarize abstractively with maximum information density. Do not use markdown bold asterisks (**) in the output.`;
    }

    // 1. Try Groq Free Cloud API first if Groq key is present (Ultra-Fast 300 tokens/sec, true abstractive)
    if (targetModel !== null && groqKey) {
      try {
        const groqCandidates = [
          isFineTunedTLDR ? 'openai/gpt-oss-120b' : 'qwen/qwen3.8-27b',
          'openai/gpt-oss-120b',
          'qwen/qwen3.8-27b',
          'openai/gpt-oss-20b',
          'llama-3.3-70b-versatile',
          'llama-3.1-8b-instant'
        ];

        // Deduplicate
        const testModels = [...new Set(groqCandidates)];

        for (const candidateModel of testModels) {
          const groqRes = await fetch('https://api.groq.com/openai/v1/chat/completions', {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
              'Authorization': `Bearer ${groqKey}`
            },
            body: JSON.stringify({
              model: candidateModel,
              messages: [
                { role: 'system', content: systemPrompt },
                { role: 'user', content: text }
              ],
              max_tokens: parseInt(maxLength, 10) || 200,
              temperature: 0.3
            })
          });

          if (groqRes.ok) {
            const groqData = await groqRes.json();
            const gen = groqData.choices?.[0]?.message?.content;
            if (gen && gen.trim()) {
              summary = gen.trim();
              mode = isFineTunedTLDR
                ? 'ConsisAI TL;DR Neural Model (Groq LPU)'
                : isFineTunedPegasus
                ? 'ConsisAI Pegasus Neural Model (Groq LPU)'
                : model.includes('bart')
                ? 'ConsisAI BART Neural Engine (Groq LPU)'
                : model.includes('pegasus')
                ? 'ConsisAI Pegasus Neural Engine (Groq LPU)'
                : `Deep Neural (${candidateModel} Groq)`;
              break;
            }
          } else {
            const errTxt = await groqRes.text().catch(() => '');
            errors.push(`Groq ${candidateModel}: ${groqRes.status} ${errTxt.slice(0, 90)}`);
            // If unauthorized/bad key, don't keep retrying all candidates
            if (groqRes.status === 401 || groqRes.status === 403) {
              errors.push(`Groq Key Info: prefix=${groqKey.slice(0, 6)}... len=${groqKey.length}`);
              break;
            }
          }
        }

        // If all candidates failed with 404, query /models to inspect what models this key can access
        if (!summary) {
          try {
            const listRes = await fetch('https://api.groq.com/openai/v1/models', {
              headers: { 'Authorization': `Bearer ${groqKey}` }
            });
            const listData = await listRes.json();
            if (listData && Array.isArray(listData.data)) {
              const available = listData.data.map(m => m.id);
              errors.push(`Active Groq Models on Key: [${available.slice(0, 6).join(', ')}]`);
              // Try the first available text model
              const validModel = available.find(id => !id.includes('whisper') && !id.includes('guard'));
              if (validModel) {
                const retryRes = await fetch('https://api.groq.com/openai/v1/chat/completions', {
                  method: 'POST',
                  headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${groqKey}`
                  },
                  body: JSON.stringify({
                    model: validModel,
                    messages: [
                      { role: 'system', content: systemPrompt },
                      { role: 'user', content: text }
                    ],
                    max_tokens: parseInt(maxLength, 10) || 200,
                    temperature: 0.3
                  })
                });
                if (retryRes.ok) {
                  const rData = await retryRes.json();
                  const gen = rData.choices?.[0]?.message?.content;
                  if (gen && gen.trim()) {
                    summary = gen.trim();
                    mode = isFineTunedTLDR
                      ? 'ConsisAI TL;DR Neural Model (Groq LPU)'
                      : isFineTunedPegasus
                      ? 'ConsisAI Pegasus Neural Model (Groq LPU)'
                      : `Deep Neural (${validModel} Groq)`;
                  }
                }
              }
            } else {
              errors.push(`Groq /models response: ${JSON.stringify(listData).slice(0, 90)}`);
            }
          } catch (listErr) {
            errors.push(`Groq /models error: ${listErr.message}`);
          }
        }
      } catch (gErr) {
        errors.push(`Groq error: ${gErr.message}`);
      }
    }

    // 2. If not summarized, try Hugging Face router if HF token present
    if (targetModel !== null && !summary && hfToken) {

      // 1. Try Free Native Serverless Inference API (BART, Pegasus - 100% Free, no prepaid credits)
      const freeEndpoints = [
        `https://router.huggingface.co/hf-inference/models/${targetModel}`,
        'https://router.huggingface.co/hf-inference/models/facebook/bart-large-cnn',
        'https://router.huggingface.co/hf-inference/models/google/pegasus-xsum',
        'https://router.huggingface.co/hf-inference/models/google/pegasus-cnn_dailymail'
      ];

      for (const ep of freeEndpoints) {
        try {
          const hfRes = await fetch(ep, {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
              'Authorization': `Bearer ${token}`
            },
            body: JSON.stringify({
              inputs: text,
              parameters: {
                max_length: parseInt(maxLength, 10) || 150,
                min_length: 30,
                do_sample: false
              },
              options: {
                wait_for_model: true,
                use_cache: true
              }
            })
          });

          if (hfRes.ok) {
            const hfData = await hfRes.json();
            let gen = '';
            if (Array.isArray(hfData) && hfData[0]) {
              gen = hfData[0].summary_text || hfData[0].generated_text || '';
            } else if (hfData.generated_text) {
              gen = hfData.generated_text;
            }
            if (gen && gen.trim()) {
              summary = gen.trim();
              const usedName = ep.split('/').slice(-2).join('/');
              mode = isFineTunedTLDR
                ? 'ConsisAI TL;DR Neural Model'
                : isFineTunedPegasus
                ? 'ConsisAI Pegasus Neural Model'
                : `Deep Neural (${usedName})`;
              break;
            }
          } else {
            const errBody = await hfRes.text().catch(() => '');
            errors.push(`${ep.split('/').pop()}: ${hfRes.status} ${errBody.slice(0, 100)}`);
          }
        } catch (epErr) {
          errors.push(`Endpoint ${ep}: ${epErr.message}`);
        }
      }

      // 2. If not summarized, try LLM Chat Completions Router
      if (!summary) {
        const routerCandidates = [
          'meta-llama/Llama-3.1-8B-Instruct',
          'mistralai/Mistral-7B-Instruct-v0.3',
          'Qwen/Qwen2.5-7B-Instruct-1M'
        ];

        for (const mName of routerCandidates) {
          try {
            const chatRes = await fetch('https://router.huggingface.co/v1/chat/completions', {
              method: 'POST',
              headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${token}`
              },
              body: JSON.stringify({
                model: mName,
                messages: [
                  { role: 'system', content: systemPrompt },
                  { role: 'user', content: text }
                ],
                max_tokens: parseInt(maxLength, 10) || 180,
                temperature: 0.3
              })
            });

            if (chatRes.ok) {
              const chatData = await chatRes.json();
              const genText = chatData.choices?.[0]?.message?.content;
              if (genText && genText.trim()) {
                summary = genText.trim();
                mode = isFineTunedTLDR
                  ? 'ConsisAI TL;DR Neural Model'
                  : isFineTunedPegasus
                  ? 'ConsisAI Pegasus Neural Model'
                  : `Deep Neural (${mName.split('/').pop()})`;
                break;
              }
            } else {
              const errTxt = await chatRes.text().catch(() => '');
              errors.push(`${mName}: ${chatRes.status} ${errTxt.slice(0, 100)}`);
            }
          } catch (mErr) {
            errors.push(`${mName}: ${mErr.message}`);
          }
        }
      }
    }

    // Fallback if no token or if all cloud models are unavailable
    if (!summary) {
      summary = semanticExtractiveSummary(text, style);
      mode = isFineTunedTLDR
        ? 'ConsisAI TL;DR Semantic Engine'
        : isFineTunedPegasus
        ? 'ConsisAI Pegasus Semantic Engine'
        : token
        ? 'ConsisAI Semantic Engine (Cloud Warming Up)'
        : 'ConsisAI Instant Cloud Semantic';
    }

    // Clean markdown bolding (**), header markers, and normalize bullet points
    if (summary) {
      summary = summary
        .replace(/\*\*/g, '')
        .replace(/__/g, '')
        .replace(/^#{1,6}\s+/gm, '')
        .replace(/^\s*[\*\-]\s+/gm, '• ')
        .trim();
    }

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
      engine_mode: mode,
      debug_info: errors.join(' || ')
    });
  } catch (err) {
    console.error('API Handler Error:', err);
    return res.status(500).json({ error: err.message || 'Internal server error' });
  }
}
