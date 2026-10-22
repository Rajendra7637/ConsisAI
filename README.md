# ConsisAI - Vercel Serverless Deployment

This directory contains the production-ready **Vercel Serverless** package for **ConsisAI**, developed by **Rajendra Singh Bisht**.

---

## Architecture Overview

Vercel Serverless Functions have a strict **250 MB package size limit**. Deep learning libraries like PyTorch (`torch` ~1.5 GB) and raw transformer checkpoints (~2 GB) exceed this limit if packaged directly.

This deployment solves that with an elegant, production-grade cloud architecture:
1. **Frontend**: Lightweight, high-performance static web app served on Vercel's global Edge CDN with deep dark cozy themes, pure white typography, sample buttons, and real-time compression analytics.
2. **Serverless API Route (`/api/summarize`)**: A micro-serverless handler (< 5 KB) executing on Vercel Node.js runtime.
3. **Neural Summarization Engine**: Routes requests to high-performance inference endpoints (Hugging Face Serverless Inference API supporting `facebook/bart-large-cnn`, `google/pegasus-xsum`, `Qwen/Qwen2.5-7B-Instruct`).

---

##  How to Deploy to Vercel

### Method 1: Deploy with Vercel CLI (Quickest)

1. Open PowerShell / Command Prompt and navigate to this folder:
   ```powershell
   cd D:\ConsisAI\vercel_deployment
   ```

2. Install the Vercel CLI (if not already installed):
   ```powershell
   npm install -g vercel
   ```

3. Deploy:
   ```powershell
   vercel
   ```
   - Follow the simple interactive prompts:
     - Set up and deploy? **Yes (`y`)**
     - Which scope? **Your account**
     - Link to existing project? **No (`n`)**
     - Project name? **`consis-ai`**
     - Directory? **`./`**
   - Vercel will give you an instant live preview URL (e.g. `https://consis-ai-xxxx.vercel.app`).

4. Deploy to Production:
   ```powershell
   vercel --prod
   ```

---

### Method 2: Deploy via GitHub & Vercel Web Dashboard (1-Click)

1. Create a new GitHub repository named `consis-ai`.
2. Push the contents of `D:\ConsisAI\vercel_deployment` to your repository:
   ```powershell
   cd D:\ConsisAI\vercel_deployment
   git init
   git add .
   git commit -m "Initial ConsisAI Vercel release"
   git branch -M main
   git remote add origin https://github.com/<your-username>/consis-ai.git
   git push -u origin main
   ```
3. Go to [https://vercel.com/new](https://vercel.com/new).
4. Click **Import** next to your `consis-ai` repository.
5. Click **Deploy**. Vercel will build and assign you a permanent public domain:
    `https://consis-ai.vercel.app`

---

## Environment Variables (Optional for High Traffic)

By default, the serverless function can use Hugging Face's public tier. For higher rate limits:
1. Go to your Project Settings on Vercel -> **Environment Variables**.
2. Add:
   - Key: `HF_TOKEN`
   - Value: Your Hugging Face user access token (`hf_...` from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)).
3. Redeploy.

---

### Developed By
**Rajendra Singh Bisht**  
ConsisAI Neural Engine
