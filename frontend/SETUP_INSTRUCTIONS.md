# 🚀 Setup Instructions for Next.js 16 Frontend

## ✅ Current Status

Your project structure is properly set up! Here's what needs to be done to get it running.

## 📦 Step 1: Install Missing Dependencies

Run this command in the `frontend` directory:

```bash
npm install
```

This will install the newly added dependencies:

- `@headlessui/react` - For accessible UI components (Tabs)
- `recharts` - For charts and data visualization
- `clsx` - For conditional CSS classes

## 🔧 Step 2: Verify Environment Variables

Make sure your `.env` file contains:

```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```

## 🏃 Step 3: Run the Development Server

```bash
npm run dev
```

The app will be available at: **http://localhost:3000**

## 🔍 What Was Fixed

### ✅ Dependencies Added to package.json

- `@headlessui/react`: ^2.2.0
- `recharts`: ^2.15.0
- `clsx`: ^2.1.1

### ✅ Icon Name Fixed

- Changed `Waveform` to `AudioWaveform` (lucide-react update)

### ✅ TypeScript Error Fixed

- Added proper type annotation for Tab render prop: `({ selected }: { selected: boolean })`

### ✅ CSS Keyframes Added

- Added `@keyframes glow` for the glowing danger effect

## 📋 Full Command Sequence

```bash
# In the frontend directory
cd "D:\ml practice\mutimodel-heart-sounds\frontend"

# Install dependencies
npm install

# Run development server
npm run dev
```

## 🎯 Next.js 16 & Tailwind CSS v4 Notes

- ✅ Using Tailwind CSS v4 with `@tailwindcss/postcss`
- ✅ Next.js 16.1.4 with React 19
- ✅ App Router structure (not Pages Router)
- ✅ PostCSS configuration updated for Tailwind v4

## 🐛 If You See Errors

### CSS Warnings (@tailwind)

The CSS linter warnings about `@tailwind` directives are expected and can be ignored. Tailwind CSS v4 handles these correctly.

### Module Not Found

If you see "Cannot find module" errors:

```bash
rm -rf node_modules package-lock.json
npm install
```

### Port Already in Use

If port 3000 is busy:

```bash
npm run dev -- -p 3001
```

## ✅ Backend Must Be Running

Before testing the frontend, ensure the FastAPI backend is running:

```bash
# In the project root directory
cd "D:\ml practice\mutimodel-heart-sounds"
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## 🎉 You're All Set!

Once you run `npm install` and `npm run dev`, the system will be ready to use.

**Access Points:**

- 🎨 Frontend: http://localhost:3000
- 🔌 Backend: http://localhost:8000
- 📚 API Docs: http://localhost:8000/docs
