import '@testing-library/jest-dom/vitest'
import { afterEach } from 'vitest'

const storageValues = new Map<string, string>()
const memoryStorage: Storage = {
  get length() { return storageValues.size },
  clear: () => storageValues.clear(),
  getItem: (key) => storageValues.get(key) ?? null,
  key: (index) => [...storageValues.keys()][index] ?? null,
  removeItem: (key) => { storageValues.delete(key) },
  setItem: (key, value) => { storageValues.set(key, String(value)) },
}

Object.defineProperty(window, 'localStorage', { configurable: true, value: memoryStorage })
afterEach(() => memoryStorage.clear())
