export class DaemonClient {
  private ws: WebSocket | null = null;
  private url: string;

  constructor(url: string = 'ws://localhost:8000/ws/client/connect') {
    this.url = url;
  }

  connect(): Promise<void> {
    return new Promise((resolve, reject) => {
      this.ws = new WebSocket(this.url);
      this.ws.onopen = () => resolve();
      this.ws.onerror = (e) => reject(e);
      this.ws.onmessage = (event) => this.handleMessage(event.data);
    });
  }

  send(method: string, params: any) {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({
        jsonrpc: '2.0',
        method,
        params,
        id: Date.now()
      }));
    }
  }

  private handleMessage(data: string) {
    try {
      const msg = JSON.parse(data);
      console.log('Received:', msg);
      // Dispatch event or callback
    } catch (e) {
      console.error('Parse error', e);
    }
  }
}
