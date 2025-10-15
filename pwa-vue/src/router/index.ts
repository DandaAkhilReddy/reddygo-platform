import { createRouter, createWebHistory, type RouteRecordRaw } from 'vue-router';
import { useAuthStore } from '../stores/auth';

// Route definitions
const routes: RouteRecordRaw[] = [
  {
    path: '/',
    name: 'dashboard',
    component: () => import('../views/DashboardView.vue'),
    meta: { requiresAuth: true },
  },
  {
    path: '/login',
    name: 'login',
    component: () => import('../views/LoginView.vue'),
    meta: { requiresAuth: false },
  },
  {
    path: '/pytorch',
    name: 'pytorch',
    component: () => import('../views/PytorchDemoView.vue'),
    meta: { requiresAuth: true },
  },
  {
    path: '/tensorflow',
    name: 'tensorflow',
    component: () => import('../views/TensorflowDemoView.vue'),
    meta: { requiresAuth: true },
  },
  {
    path: '/qdrant',
    name: 'qdrant',
    component: () => import('../views/QdrantDemoView.vue'),
    meta: { requiresAuth: true },
  },
  {
    path: '/langchain',
    name: 'langchain',
    component: () => import('../views/LangchainChatView.vue'),
    meta: { requiresAuth: true },
  },
  {
    path: '/cost-analysis',
    name: 'cost-analysis',
    component: () => import('../views/CostAnalysisView.vue'),
    meta: { requiresAuth: true },
  },
  {
    path: '/:pathMatch(.*)*',
    name: 'not-found',
    redirect: '/',
  },
];

const router = createRouter({
  history: createWebHistory(),
  routes,
});

// Navigation guard for authentication
router.beforeEach((to, _from, next) => {
  const authStore = useAuthStore();
  const requiresAuth = to.meta.requiresAuth;

  if (requiresAuth && !authStore.isAuthenticated) {
    // Redirect to login if not authenticated
    next({ name: 'login', query: { redirect: to.fullPath } });
  } else if (to.name === 'login' && authStore.isAuthenticated) {
    // Redirect to dashboard if already authenticated
    next({ name: 'dashboard' });
  } else {
    next();
  }
});

export default router;
