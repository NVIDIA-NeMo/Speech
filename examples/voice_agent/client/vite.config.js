import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react-swc';

export default defineConfig({
    plugins: [react()],
    server: {
        host: '0.0.0.0',  // Bind to all interfaces
        port: 5173,
        proxy: {
            // Proxy /connect POST to the FastAPI backend
            '/connect': {
                target: 'http://127.0.0.1:7860',
                changeOrigin: true,
            },
            // Proxy /ws WebSocket connections to the pipecat WebSocket server
            '/ws': {
                target: 'ws://127.0.0.1:8765',
                ws: true,
                changeOrigin: true,
                rewrite: (path) => path.replace(/^\/ws/, ''),
            },
        },
    },
});
