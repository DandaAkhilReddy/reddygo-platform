import { createApp } from 'vue';
import { createPinia } from 'pinia';
import App from './App.vue';
import router from './router';
import { firebaseAuth } from './services/firebase';
import { useAuthStore } from './stores/auth';
import './style.css';

// Create Vue app
const app = createApp(App);

// Create Pinia store
const pinia = createPinia();
app.use(pinia);

// Use router
app.use(router);

// Initialize Firebase auth state observer
const authStore = useAuthStore();
firebaseAuth.onAuthStateChange((user) => {
  authStore.setUser(user);
});

// Mount app
app.mount('#app');
