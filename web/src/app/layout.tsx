import type { Metadata } from 'next';
import { Inter } from 'next/font/google';
import './globals.css';

const inter = Inter({ subsets: ['latin'], variable: '--font-inter' });

export const metadata: Metadata = {
  title: 'MANDIMITRA | AI-Powered Agricultural Intelligence',
  description: 'Empowering Indian Farmers with AI-driven crop risk assessment and price intelligence for smarter farming decisions.',
  keywords: ['agriculture', 'farming', 'crop risk', 'price prediction', 'AI', 'machine learning', 'India', 'mandi'],
  authors: [{ name: 'MANDIMITRA Team' }],
  openGraph: {
    title: 'MANDIMITRA | AI-Powered Agricultural Intelligence',
    description: 'Empowering Indian Farmers with AI-driven crop risk assessment and price intelligence',
    type: 'website',
  },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className={`${inter.variable} font-sans antialiased`}>
        {children}
      </body>
    </html>
  );
}
