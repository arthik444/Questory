import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.tsx'

// Note: StrictMode removed — it causes intentional double-mount in dev which
// creates two simultaneous WebSocket sessions (two voices) for Live API hooks.
createRoot(document.getElementById('root')!).render(<App />)
