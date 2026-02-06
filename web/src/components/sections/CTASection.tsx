'use client';

import { motion } from 'framer-motion';
import { ArrowRight, Sparkles } from 'lucide-react';
import Link from 'next/link';

export function CTASection() {
  return (
    <section className="py-24 bg-gradient-to-br from-slate-900 via-emerald-900 to-slate-900 relative overflow-hidden">
      {/* Background Elements */}
      <div className="absolute inset-0 bg-grid-pattern opacity-5" />
      <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-emerald-500/20 rounded-full blur-3xl" />
      <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-amber-500/20 rounded-full blur-3xl" />

      <div className="relative max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
        >
          {/* Badge */}
          <div className="inline-flex items-center px-4 py-2 bg-emerald-500/20 text-emerald-300 rounded-full text-sm font-medium mb-6 border border-emerald-500/30">
            <Sparkles className="w-4 h-4 mr-2" />
            Start Your Free Trial Today
          </div>

          {/* Heading */}
          <h2 className="text-3xl sm:text-4xl lg:text-5xl font-bold text-white mb-6">
            Ready to Transform Your
            <br />
            <span className="bg-gradient-to-r from-emerald-400 to-amber-400 bg-clip-text text-transparent">
              Farming Journey?
            </span>
          </h2>

          {/* Description */}
          <p className="text-lg text-slate-300 mb-8 max-w-2xl mx-auto">
            Join thousands of farmers who are already making smarter decisions with MANDIMITRA. 
            No credit card required. Get started in 2 minutes.
          </p>

          {/* CTA Buttons */}
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Link
              href="/crop-risk"
              className="group inline-flex items-center justify-center px-8 py-4 bg-gradient-to-r from-emerald-500 to-emerald-400 text-white font-semibold rounded-xl hover:from-emerald-600 hover:to-emerald-500 transition-all shadow-lg shadow-emerald-500/30 hover:shadow-emerald-500/50 hover:scale-105"
            >
              Try Crop Risk Advisor
              <ArrowRight className="ml-2 w-5 h-5 group-hover:translate-x-1 transition-transform" />
            </Link>
            <Link
              href="/price-forecast"
              className="group inline-flex items-center justify-center px-8 py-4 bg-white/10 text-white font-semibold rounded-xl border border-white/20 hover:bg-white/20 transition-all hover:scale-105 backdrop-blur-sm"
            >
              Explore Price Forecast
            </Link>
          </div>

          {/* Trust Text */}
          <p className="mt-8 text-sm text-slate-400">
            🔒 Your data is secure and private. We never share your information.
          </p>
        </motion.div>
      </div>
    </section>
  );
}
