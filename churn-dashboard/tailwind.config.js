export default {
  content: ['./index.html', './src/**/*.{js,jsx,ts,tsx}'],
  darkMode: 'class',
  theme: {
    extend: {
      boxShadow: {
        soft: '0 24px 80px rgba(15, 23, 42, 0.08)',
      },
      backgroundImage: {
        'hero-radial': 'radial-gradient(circle at top left, rgba(59, 130, 246, 0.18), transparent 30%), radial-gradient(circle at bottom right, rgba(16, 185, 129, 0.12), transparent 28%)',
      },
    },
  },
  plugins: [],
}
