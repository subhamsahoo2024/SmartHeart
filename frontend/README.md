# ❤️ SmartHeart - Frontend

A modern, responsive web application for multi-modal cardiovascular health analysis using ECG (Electrocardiogram) and PCG (Phonocardiogram) signals. Built with Next.js 16, React 19, and Tailwind CSS v4.

## 🚀 Overview

SmartHeart provides an intuitive interface for healthcare professionals and researchers to analyze heart signals using advanced machine learning models. The application supports:

- **ECG Analysis**: Detects arrhythmias and abnormal heart rhythms
- **PCG Analysis**: Identifies heart murmurs and valve abnormalities
- **Multi-Modal Fusion**: Combined analysis for comprehensive cardiovascular assessment
- **Real-time Visualization**: Interactive charts, spectrograms, and waveforms
- **Risk Scoring**: Confidence-based risk assessment with visual indicators

## ✨ Features

### 📤 File Upload

- Drag-and-drop or click-to-upload interface
- Support for ECG (.csv) and PCG (.wav) files
- Real-time validation and error handling
- Single or dual-modality analysis

### 📊 Visualization Components

- **Signal Charts**: Time-domain waveform visualization with Recharts
- **Spectrograms**: Frequency-domain PCG analysis with interactive heatmaps
- **Risk Bar Charts**: Confidence scores with color-coded risk levels
- **Tabbed Interface**: Organized views for ECG, PCG, and combined results

### 🎨 Modern UI/UX

- Dark mode with gradient accents
- Smooth animations with Framer Motion
- Responsive design (mobile, tablet, desktop)
- Loading states and error handling
- Accessible components with Headless UI

## 🛠️ Tech Stack

### Core Framework

- **Next.js 16.1.4** - App Router with React Server Components
- **React 19.2.3** - Latest React with concurrent features
- **TypeScript 5** - Type-safe development

### Styling & UI

- **Tailwind CSS v4** - Utility-first CSS framework
- **@tailwindcss/postcss** - Latest PostCSS integration
- **clsx** - Conditional class names
- **tailwind-merge** - Smart class merging

### Data Visualization

- **Recharts 2.15.4** - Composable charting library
- **Custom Chart Components** - SignalChart, SpectrogramViewer, RiskBarChart

### UI Components

- **@headlessui/react 2.2.0** - Unstyled, accessible components
- **Lucide React 0.562.0** - Modern icon library
- **Framer Motion 12.27.5** - Animation library

### HTTP & API

- **Axios 1.13.2** - Promise-based HTTP client
- API service layer with TypeScript interfaces

### Utilities

- **html2canvas 1.4.1** - Export visualizations
- **jsPDF 4.0.0** - Generate PDF reports

## 📦 Installation

### Prerequisites

- Node.js 20+ (LTS recommended)
- npm, yarn, pnpm, or bun

### Setup

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Create environment file
cp .env.example .env
```

### Environment Variables

Create a `.env` file in the frontend directory:

```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```

## 🏃 Running the Application

### Development Mode

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) to view the application.

The page auto-reloads when you make changes. You can start editing by modifying [app/page.tsx](app/page.tsx).

### Production Build

```bash
# Build the application
npm run build

# Start production server
npm start
```

### Linting

```bash
npm run lint
```

## 📁 Project Structure

```
frontend/
├── app/                    # Next.js App Router
│   ├── layout.tsx         # Root layout with metadata
│   ├── page.tsx           # Main application page
│   └── globals.css        # Global styles & Tailwind directives
├── components/
│   ├── charts/            # Visualization components
│   │   ├── SignalChart.tsx        # Time-domain waveform viewer
│   │   ├── SpectrogramViewer.tsx  # Frequency-domain heatmap
│   │   └── RiskBarChart.tsx       # Confidence score display
│   └── ui/                # Reusable UI components
│       ├── Card.tsx       # Container with gradient border
│       └── Tabs.tsx       # Accessible tab navigation
├── services/
│   └── api.ts             # API client & TypeScript interfaces
├── lib/
│   └── utils.ts           # Utility functions (cn, etc.)
├── public/                # Static assets
└── next.config.ts         # Next.js configuration
```

## 🔌 API Integration

The frontend communicates with the backend API via Axios. The API service is located in [services/api.ts](services/api.ts).

### Key Endpoints

```typescript
POST /predict
- Accepts: multipart/form-data
- Fields: ecg_file (optional), pcg_file (optional)
- Returns: PredictionResponse with risk scores and visualization data
```

### Response Interface

```typescript
interface PredictionResponse {
  ecg_risk: number | null; // 0-1 confidence score
  pcg_risk: number | null; // 0-1 confidence score
  combined_risk: number | null; // Weighted average
  ecg_label: string | null; // "Normal" or "Abnormal"
  pcg_label: string | null; // "Normal" or "Abnormal"
  combined_label: string | null; // Final classification
  ecg_plot_data: [number, number][]; // [(time, voltage)]
  pcg_waveform_data: number[]; // Amplitude array
  pcg_spectrogram_data: number[][]; // 2D frequency matrix
}
```

## 🎨 Styling Guidelines

### Tailwind CSS v4

This project uses the latest Tailwind CSS v4 with PostCSS integration. Key features:

- **New @import syntax**: Uses `@import "tailwindcss"` in globals.css
- **Simplified configuration**: No tailwind.config.js needed
- **Performance improvements**: Faster builds and smaller bundles

### Color Palette

```css
Primary: emerald-400 to emerald-600
Secondary: blue-400 to blue-600
Success: emerald-500
Warning: yellow-500
Danger: red-400 to red-600
Background: slate-900 to slate-950
Text: white, gray-300
```

### Custom Animations

- Pulse effects for loading states
- Glow effects for high-risk indicators
- Smooth transitions with Framer Motion

## 🧪 Development Tips

### Hot Module Replacement

Next.js supports HMR out of the box. Changes to components will hot-reload without full page refresh.

### TypeScript Strict Mode

The project uses strict TypeScript. Ensure all props and state have proper types.

### Linting Rules

ESLint is configured with Next.js recommended rules. Run `npm run lint` before committing.

### Component Best Practices

- Use client components (`"use client"`) for interactive features
- Keep server components for static content
- Extract reusable logic into custom hooks
- Use TypeScript interfaces for all props

## 🐛 Troubleshooting

### Port Already in Use

```bash
# Use a different port
npm run dev -- -p 3001
```

### Module Not Found

```bash
# Clear node_modules and reinstall
rm -rf node_modules package-lock.json
npm install
```

### CSS @tailwind Warnings

The linter may show warnings about `@tailwind` directives. These are expected with Tailwind CSS v4 and can be safely ignored.

### Backend Connection Errors

Ensure the backend API is running on `http://localhost:8000`. Check the `.env` file for the correct `NEXT_PUBLIC_API_URL`.

## 📚 Learn More

### Next.js Resources

- [Next.js Documentation](https://nextjs.org/docs) - Learn about Next.js features and API
- [Next.js App Router](https://nextjs.org/docs/app) - Deep dive into App Router
- [React Server Components](https://nextjs.org/docs/app/building-your-application/rendering/server-components)

### Tailwind CSS v4

- [Tailwind CSS v4 Alpha Docs](https://tailwindcss.com/docs)
- [Tailwind CSS v4 Migration Guide](https://tailwindcss.com/docs/upgrade-guide)

### Visualization Libraries

- [Recharts Documentation](https://recharts.org/en-US/)
- [Framer Motion Docs](https://www.framer.com/motion/)

## 🚀 Deployment

### Vercel (Recommended)

The easiest way to deploy is using [Vercel Platform](https://vercel.com/new):

1. Push your code to GitHub
2. Import the repository in Vercel
3. Add environment variables
4. Deploy!

Vercel automatically detects Next.js and configures build settings.

### Docker

```dockerfile
FROM node:20-alpine
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production
COPY . .
RUN npm run build
EXPOSE 3000
CMD ["npm", "start"]
```

### Environment Variables

Remember to set `NEXT_PUBLIC_API_URL` in your deployment environment to point to your production backend API.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is part of the SmartHeart multi-modal cardiovascular analysis system.

## 🔗 Related

- [Backend API Documentation](../backend/README.md)
- [Setup Instructions](SETUP_INSTRUCTIONS.md)
