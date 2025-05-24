// vite.config.js
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite'


console.log('🐶  Vite config is loading!')

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    proxy: {
      // forward /api/* to FastAPI
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        secure: false
      },
      // forward /static/* to FastAPI's /static/* mount
      '/static': {
        target: 'http://localhost:8000/static',
        changeOrigin: true,
        secure: false,
        // rewrite: (path) => path.replace(/^\/static/, '')
      }
    }
  }
});