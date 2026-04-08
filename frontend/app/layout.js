import './globals.css';

export const metadata = {
  title: 'Drishti AI Command Center',
  description: 'Real-time crowd intelligence and safety monitoring platform powered by AI agents',
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
