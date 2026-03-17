/**
 * VltBridge — PostMessage API for artifact iframes.
 *
 * Injected into every artifact iframe by the daemon. Provides access to
 * platform capabilities (storage, notes, code search, backend, connectors,
 * events) via postMessage to the parent page.
 *
 * All methods return Promises. The parent page handles routing to daemon APIs.
 */
(function () {
  'use strict';

  var _requestId = 0;
  var _pending = {};  // id -> {resolve, reject}
  var _eventHandlers = {};  // event_type -> [callback]

  // Listen for responses from parent
  window.addEventListener('message', function (e) {
    var msg = e.data;
    if (!msg || typeof msg !== 'object') return;

    if (msg.type === 'vlt_response' && msg.id in _pending) {
      var handler = _pending[msg.id];
      delete _pending[msg.id];
      if (!msg.ok || msg.error) {
        var errMsg = typeof msg.error === 'string' ? msg.error
          : (msg.error && (msg.error.message || msg.error.code)) || 'Unknown error';
        handler.reject(new Error(errMsg));
      } else {
        handler.resolve(msg.result);
      }
    }

    if (msg.type === 'vlt_event') {
      var handlers = _eventHandlers[msg.event_type] || [];
      var wildcards = _eventHandlers['*'] || [];
      handlers.concat(wildcards).forEach(function (cb) {
        try { cb(msg); } catch (err) { console.error('Event handler error:', err); }
      });
    }
  });

  function _request(method, params) {
    return new Promise(function (resolve, reject) {
      var id = 'req-' + (++_requestId);
      _pending[id] = { resolve: resolve, reject: reject };

      // Timeout after 30s
      setTimeout(function () {
        if (id in _pending) {
          delete _pending[id];
          reject(new Error('VltBridge request timed out: ' + method));
        }
      }, 30000);

      window.parent.postMessage({
        type: 'vlt_request',
        id: id,
        method: method,
        params: params || {}
      }, '*');
    });
  }

  window.VltBridge = {
    // Artifact-scoped key-value storage
    storage: {
      get: function (key) { return _request('storage.get', { key: key }); },
      set: function (key, value) { return _request('storage.set', { key: key, value: value }); },
      list: function () { return _request('storage.list', {}); }
    },

    // Vault note access
    notes: {
      read: function (path) { return _request('notes.read', { path: path }); },
      write: function (path, content) { return _request('notes.write', { path: path, content: content }); },
      search: function (query) { return _request('notes.search', { query: query }); }
    },

    // Code search
    code: {
      search: function (query) { return _request('code.search', { query: query }); },
      map: function () { return _request('code.map', {}); }
    },

    // Backend process calls
    backend: {
      call: function (action, params) {
        return _request('backend.call', { action: action, params: params || {} });
      }
    },

    // Connector access (proxied, no raw creds)
    connectors: {
      list: function () { return _request('connectors.list', {}); },
      call: function (connector, instance, action, params) {
        return _request('connectors.call', {
          connector: connector,
          instance: instance,
          action: action,
          params: params || {}
        });
      }
    },

    // Artifact-to-artifact IPC events
    events: {
      emit: function (eventType, payload) {
        return _request('events.emit', { event_type: eventType, payload: payload || {} });
      },
      on: function (eventType, callback) {
        if (!_eventHandlers[eventType]) _eventHandlers[eventType] = [];
        _eventHandlers[eventType].push(callback);
        return function unsubscribe() {
          _eventHandlers[eventType] = _eventHandlers[eventType].filter(function (cb) {
            return cb !== callback;
          });
        };
      }
    },

    // Artifact self-management
    artifact: {
      getState: function () { return _request('artifact.getState', {}); },
      setState: function (state) { return _request('artifact.setState', { state: state }); },
      log: function (message) { return _request('artifact.log', { message: message }); }
    }
  };

  // HMR state preservation hooks (set by artifact code)
  window.__vlt_save_state = null;
  window.__vlt_load_state = null;
})();
