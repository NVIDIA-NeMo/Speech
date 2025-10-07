import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react-swc';

export default defineConfig({
    plugins: [react()],
    server: {
        host: '0.0.0.0',  // Bind to all interfaces
        port: 5173,       // Back to default Vite port
        proxy: {
            // Proxy /api requests to the backend server
            '/connect': {
                target: 'http://0.0.0.0:7860', // Replace with your backend URL if needed
                changeOrigin: true,
            },
        },
    },
});
