/**
 * Copyright (c) 2024–2025, Daily
 *
 * SPDX-License-Identifier: BSD 2-Clause License
 */

/**
 * RTVI Client Implementation
 *
 * This client connects to an RTVI-compatible bot server using WebSocket.
 *
 * Requirements:
 * - A running RTVI bot server (defaults to http://localhost:7860)
 */

import {
  RTVIClient,
  RTVIClientOptions,
  RTVIEvent,
} from '@pipecat-ai/client-js';
import {
  WebSocketTransport
} from "@pipecat-ai/websocket-transport";

class WebsocketClientApp {
  private rtviClient: RTVIClient | null = null;
  private connectBtn: HTMLButtonElement | null = null;
  private disconnectBtn: HTMLButtonElement | null = null;
  private muteBtn: HTMLButtonElement | null = null;
  private resetBtn: HTMLButtonElement | null = null;
  private serverSelect: HTMLSelectElement | null = null;
  private statusSpan: HTMLElement | null = null;
  private debugLog: HTMLElement | null = null;
  private volumeBar: HTMLElement | null = null;
  private volumeText: HTMLElement | null = null;
  private botAudio: HTMLAudioElement;
  private isConnecting: boolean = false;
  private isDisconnecting: boolean = false;
  private isMuted: boolean = false;
  private audioContext: AudioContext | null = null;
  private analyser: AnalyserNode | null = null;
  private microphone: MediaStreamAudioSourceNode | null = null;
  private volumeUpdateInterval: number | null = null;

  // Server configurations
  private readonly serverConfigs = {
    websocket: {
      name: 'WebSocket Server',
      baseUrl: `http://${window.location.hostname}:7860`,
      port: 8765
    },
    fastapi: {
      name: 'FastAPI Server', 
      baseUrl: `http://${window.location.hostname}:8000`,
      port: 8000
    }
  };

  constructor() {
    console.log("WebsocketClientApp");
    this.botAudio = document.createElement('audio');
    this.botAudio.autoplay = true;
    //this.botAudio.playsInline = true;
    document.body.appendChild(this.botAudio);

    this.setupDOMElements();
    this.setupEventListeners();
  }

  /**
   * Set up references to DOM elements and create necessary media elements
   */
  private setupDOMElements(): void {
    this.connectBtn = document.getElementById('connect-btn') as HTMLButtonElement;
    this.disconnectBtn = document.getElementById('disconnect-btn') as HTMLButtonElement;
    this.muteBtn = document.getElementById('mute-btn') as HTMLButtonElement;
    this.resetBtn = document.getElementById('reset-btn') as HTMLButtonElement;
    this.serverSelect = document.getElementById('server-select') as HTMLSelectElement;
    this.statusSpan = document.getElementById('connection-status');
    this.debugLog = document.getElementById('debug-log');
    this.volumeBar = document.getElementById('volume-bar');
    this.volumeText = document.getElementById('volume-text');
  }

  /**
   * Set up event listeners for connect/disconnect buttons
   */
  private setupEventListeners(): void {
    this.connectBtn?.addEventListener('click', () => this.connect());
    this.disconnectBtn?.addEventListener('click', () => this.disconnect());
    this.muteBtn?.addEventListener('click', () => this.toggleMute());
    this.resetBtn?.addEventListener('click', () => this.reset());
    this.serverSelect?.addEventListener('change', () => this.updateServerUrl());
  }

  /**
   * Add a timestamped message to the debug log
   */
  private log(message: string): void {
    if (!this.debugLog) return;
    const entry = document.createElement('div');
    entry.textContent = `${new Date().toISOString()} - ${message}`;
    if (message.startsWith('User: ')) {
      entry.style.color = '#2196F3';
    } else if (message.startsWith('Bot: ')) {
      entry.style.color = '#4CAF50';
    }
    this.debugLog.appendChild(entry);
    this.debugLog.scrollTop = this.debugLog.scrollHeight;
    console.log(message);
  }

  /**
   * Update the connection status display
   */
  private updateStatus(status: string): void {
    if (this.statusSpan) {
      this.statusSpan.textContent = status;
    }
    this.log(`Status: ${status}`);
  }

  /**
   * Check for available media tracks and set them up if present
   * This is called when the bot is ready or when the transport state changes to ready
   */
  setupMediaTracks() {
    if (!this.rtviClient) return;
    const tracks = this.rtviClient.tracks();
    if (tracks.bot?.audio) {
      this.setupAudioTrack(tracks.bot.audio);
    }
  }

  /**
   * Set up listeners for track events (start/stop)
   * This handles new tracks being added during the session
   */
  setupTrackListeners() {
    if (!this.rtviClient) {
      this.log('Cannot setup track listeners: client is null');
      return;
    }

    try {
      // Listen for new tracks starting
      this.rtviClient.on(RTVIEvent.TrackStarted, (track, participant) => {
        // Only handle non-local (bot) tracks
        if (!participant?.local && track.kind === 'audio') {
          this.setupAudioTrack(track);
        }
      });

      // Listen for tracks stopping
      this.rtviClient.on(RTVIEvent.TrackStopped, (track, participant) => {
        this.log(`Track stopped: ${track.kind} from ${participant?.name || 'unknown'}`);
      });
    } catch (error) {
      this.log(`Error setting up track listeners: ${error}`);
    }
  }

  /**
   * Set up an audio track for playback
   * Handles both initial setup and track updates
   */
  private setupAudioTrack(track: MediaStreamTrack): void {
    this.log('Setting up audio track');
    if (this.botAudio.srcObject && "getAudioTracks" in this.botAudio.srcObject) {
      const oldTrack = this.botAudio.srcObject.getAudioTracks()[0];
      if (oldTrack?.id === track.id) return;
    }
    this.botAudio.srcObject = new MediaStream([track]);
  }

  /**
   * Initialize and connect to the bot
   * This sets up the RTVI client, initializes devices, and establishes the connection
   */
  public async connect(): Promise<void> {
    if (this.isConnecting) {
      this.log('Connection already in progress, ignoring...');
      return;
    }

    try {
      this.isConnecting = true;
      const startTime = Date.now();

      //const transport = new DailyTransport();
      const transport = new WebSocketTransport();
      const RTVIConfig: RTVIClientOptions = {
        transport,
        params: {
          // The baseURL and endpoint of your bot server that the client will connect to
          baseUrl: this.getSelectedServerConfig().baseUrl,
          endpoints: { connect: '/connect' },
        },
        enableMic: true,
        enableCam: false,
        callbacks: {
          onConnected: () => {
            this.updateStatus('Connected');
            if (this.connectBtn) this.connectBtn.disabled = true;
            if (this.disconnectBtn) this.disconnectBtn.disabled = false;
            if (this.muteBtn) {
              this.muteBtn.disabled = false;
              this.muteBtn.textContent = 'Mute';
            }
            if (this.resetBtn) this.resetBtn.disabled = false;
            if (this.serverSelect) this.serverSelect.disabled = true;
            // Start volume monitoring when connected
            if (!this.isMuted) {
              this.startVolumeMonitoring();
            }
          },
          onDisconnected: () => {
            // Only handle disconnect if we're not in the middle of error cleanup
            if (!this.isConnecting) {
              this.updateStatus('Disconnected');
              if (this.connectBtn) this.connectBtn.disabled = false;
              if (this.disconnectBtn) this.disconnectBtn.disabled = true;
              if (this.muteBtn) {
                this.muteBtn.disabled = true;
                this.muteBtn.textContent = 'Mute';
              }
              if (this.resetBtn) this.resetBtn.disabled = true;
              if (this.serverSelect) this.serverSelect.disabled = false;
              // Stop volume monitoring when disconnected
              this.stopVolumeMonitoring();
              this.log('Client disconnected');
            }
          },
          onBotReady: (data) => {
            this.log(`Bot ready: ${JSON.stringify(data)}`);
            this.setupMediaTracks();
          },
          onUserTranscript: (data) => {
            if (data.final) {
              this.log(`User: ${data.text}`);
            }
          },
          onBotTranscript: (data) => this.log(`Bot: ${data.text}`),
          onMessageError: (error) => console.error('Message error:', error),
          onError: (error) => console.error('Error:', error),
        },
      }
      
      // Create the client with error handling
      try {
        this.rtviClient = new RTVIClient(RTVIConfig);
        this.setupTrackListeners();
      } catch (clientError) {
        this.log(`Error creating RTVI client: ${clientError}`);
        throw clientError;
      }

      this.log('Initializing devices...');
      await this.rtviClient.initDevices();
      this.log('Devices initialized successfully');

      this.log('Connecting to bot...');
      await this.rtviClient.connect();

      const timeTaken = Date.now() - startTime;
      this.log(`Connection complete, timeTaken: ${timeTaken}`);
    } catch (error) {
      this.log(`Error connecting: ${(error as Error).message}`);
      this.updateStatus('Error');
      // Clean up if there's an error
      await this.cleanupOnError();
    } finally {
      this.isConnecting = false;
    }
  }

  /**
   * Clean up resources when there's an error during connection
   */
  private async cleanupOnError(): Promise<void> {
    // Set disconnecting flag to prevent onDisconnected callback interference
    this.isDisconnecting = true;
    
    // Store reference to client before it might become null
    const client = this.rtviClient;
    
    if (client) {
      try {
        // Check if the client is in a state where disconnect can be called
        if (typeof client.disconnect === 'function') {
          await client.disconnect();
        }
      } catch (disconnectError) {
        this.log(`Error during cleanup disconnect: ${disconnectError}`);
      } finally {
        // Always reset the client to null to allow reconnection
        this.rtviClient = null;
      }
    } else {
      this.log('Client was already null during cleanup');
    }
    
    // Reset button states
    if (this.connectBtn) this.connectBtn.disabled = false;
    if (this.disconnectBtn) this.disconnectBtn.disabled = true;
    if (this.muteBtn) {
      this.muteBtn.disabled = true;
      this.muteBtn.textContent = 'Mute';
    }
    if (this.resetBtn) this.resetBtn.disabled = true;
    if (this.serverSelect) this.serverSelect.disabled = false;
    
    // Stop volume monitoring
    this.stopVolumeMonitoring();
    
    // Reset mute state
    this.isMuted = false;
    
    // Reset disconnecting flag
    this.isDisconnecting = false;
  }

  /**
   * Disconnect from the bot and clean up media resources
   */
  public async disconnect(): Promise<void> {
    if (this.isDisconnecting) {
      this.log('Disconnection already in progress, ignoring...');
      return;
    }

    this.isDisconnecting = true;
    
    // Store reference to client before it might become null
    const client = this.rtviClient;
    
    if (client) {
      try {
        // Check if the client is in a state where disconnect can be called
        if (typeof client.disconnect === 'function') {
          await client.disconnect();
        }
      } catch (error) {
        this.log(`Error disconnecting: ${(error as Error).message}`);
      } finally {
        // Always clean up resources and reset the client
        this.rtviClient = null;
        if (this.botAudio.srcObject && "getAudioTracks" in this.botAudio.srcObject) {
          this.botAudio.srcObject.getAudioTracks().forEach((track) => track.stop());
          this.botAudio.srcObject = null;
        }
      }
    } else {
      this.log('Client was already null during disconnect');
    }
    
    // Stop volume monitoring
    this.stopVolumeMonitoring();
    
    // Reset mute state
    this.isMuted = false;
    
    this.isDisconnecting = false;
  }

  /**
   * Toggle microphone mute/unmute
   */
  private toggleMute(): void {
    if (!this.rtviClient) {
      this.log('Cannot toggle mute: client is null');
      return;
    }

    this.isMuted = !this.isMuted;
    this.rtviClient.enableMic(!this.isMuted);
    
    // Update button text
    if (this.muteBtn) {
      this.muteBtn.textContent = this.isMuted ? 'Unmute' : 'Mute';
    }
    
    // Update volume monitoring
    if (this.isMuted) {
      this.stopVolumeMonitoring();
    } else {
      this.startVolumeMonitoring();
    }
    
    this.log(this.isMuted ? 'Microphone muted' : 'Microphone unmuted');
  }

  /**
   * Start monitoring microphone volume
   */
  private async startVolumeMonitoring(): Promise<void> {
    try {
      if (!this.audioContext) {
        this.audioContext = new AudioContext();
      }

      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      
      this.analyser = this.audioContext.createAnalyser();
      this.analyser.fftSize = 256;
      this.analyser.smoothingTimeConstant = 0.8;
      
      this.microphone = this.audioContext.createMediaStreamSource(stream);
      this.microphone.connect(this.analyser);
      
      // Start continuous volume updates
      this.volumeUpdateInterval = window.setInterval(() => {
        this.updateVolumeDisplay();
      }, 100); // Update every 100ms
      
      this.log('Volume monitoring started');
    } catch (error) {
      this.log(`Error starting volume monitoring: ${error}`);
    }
  }

  /**
   * Stop monitoring microphone volume
   */
  private stopVolumeMonitoring(): void {
    if (this.volumeUpdateInterval) {
      clearInterval(this.volumeUpdateInterval);
      this.volumeUpdateInterval = null;
    }
    
    if (this.microphone) {
      this.microphone.disconnect();
      this.microphone = null;
    }
    
    // Reset volume display
    this.updateVolumeDisplay(0);
    this.log('Volume monitoring stopped');
  }

  /**
   * Update the volume display
   */
  private updateVolumeDisplay(volume?: number): void {
    if (!this.volumeBar || !this.volumeText) return;

    if (volume === undefined && this.analyser) {
      const dataArray = new Uint8Array(this.analyser.frequencyBinCount);
      this.analyser.getByteFrequencyData(dataArray);
      
      // Calculate average volume
      const average = dataArray.reduce((sum, value) => sum + value, 0) / dataArray.length;
      volume = (average / 255) * 100;
    }

    const displayVolume = volume || 0;
    const clampedVolume = Math.min(100, Math.max(0, displayVolume));
    
    this.volumeBar.style.width = `${clampedVolume}%`;
    this.volumeText.textContent = `${Math.round(clampedVolume)}%`;
    
    // Update color based on volume level
    if (clampedVolume < 30) {
      this.volumeBar.style.background = '#4caf50'; // Green
    } else if (clampedVolume < 70) {
      this.volumeBar.style.background = '#ff9800'; // Orange
    } else {
      this.volumeBar.style.background = '#f44336'; // Red
    }
  }

  /**
   * Reset the conversation context by calling the server action
   */
  private async reset(): Promise<void> {
    if (!this.rtviClient) {
      this.log('Cannot reset: not connected to server');
      return;
    }

    try {
      this.log('Resetting conversation context...');
      
      // Call the reset action on the server
      const result = await this.rtviClient.action({ service: 'context', action: 'reset', arguments: [] });
      
      if (result) {
        this.log('Conversation context reset successfully');
      } else {
        this.log('Failed to reset conversation context');
      }
    } catch (error) {
      this.log(`Error resetting context: ${error}`);
    }
  }

  private getSelectedServerConfig(): { name: string; baseUrl: string; port: number } {
    const selectedValue = this.serverSelect?.value || 'websocket';
    return this.serverConfigs[selectedValue as keyof typeof this.serverConfigs];
  }

  private updateServerUrl(): void {
    const selectedConfig = this.getSelectedServerConfig();
    this.log(`Server changed to: ${selectedConfig.name} (${selectedConfig.baseUrl})`);
    
    // If connected, show a message that they need to reconnect
    if (this.rtviClient) {
      this.log('Please disconnect and reconnect to use the new server');
    }
  }
}

declare global {
  interface Window {
    WebsocketClientApp: typeof WebsocketClientApp;
  }
}

window.addEventListener('DOMContentLoaded', () => {
  window.WebsocketClientApp = WebsocketClientApp;
  new WebsocketClientApp();
});
