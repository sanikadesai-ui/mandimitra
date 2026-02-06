'use client';

import { motion } from 'framer-motion';
import { useInView } from 'framer-motion';
import { useRef, useEffect, useState } from 'react';

const stats = [
  { value: 3500000, suffix: '+', label: 'Price Records', format: true },
  { value: 400, suffix: '+', label: 'Markets Covered', format: false },
  { value: 93, suffix: '%', label: 'Forecast Accuracy', format: false },
  { value: 36, suffix: '+', label: 'Districts', format: false },
];

function AnimatedCounter({ value, suffix, format }: { value: number; suffix: string; format: boolean }) {
  const [count, setCount] = useState(0);
  const ref = useRef(null);
  const isInView = useInView(ref, { once: true });

  useEffect(() => {
    if (isInView) {
      const duration = 2000;
      const steps = 60;
      const stepValue = value / steps;
      let current = 0;

      const timer = setInterval(() => {
        current += stepValue;
        if (current >= value) {
          setCount(value);
          clearInterval(timer);
        } else {
          setCount(Math.floor(current));
        }
      }, duration / steps);

      return () => clearInterval(timer);
    }
  }, [isInView, value]);

  const displayValue = format 
    ? (count / 1000000).toFixed(1) + 'M' 
    : count.toLocaleString();

  return (
    <span ref={ref}>
      {displayValue}{suffix}
    </span>
  );
}

export function StatsSection() {
  return (
    <section className="relative py-16 bg-gradient-to-r from-emerald-600 via-emerald-500 to-emerald-600 overflow-hidden">
      {/* Background Pattern */}
      <div className="absolute inset-0 bg-grid-pattern opacity-10" />
      
      {/* Floating Shapes */}
      <div className="absolute top-0 left-1/4 w-64 h-64 bg-white/10 rounded-full blur-3xl" />
      <div className="absolute bottom-0 right-1/4 w-64 h-64 bg-amber-500/20 rounded-full blur-3xl" />

      <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-8">
          {stats.map((stat, index) => (
            <motion.div
              key={stat.label}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: index * 0.1 }}
              className="text-center"
            >
              <div className="text-3xl sm:text-4xl lg:text-5xl xl:text-6xl font-bold text-white mb-2">
                <AnimatedCounter value={stat.value} suffix={stat.suffix} format={stat.format} />
              </div>
              <div className="text-emerald-100 text-sm sm:text-base font-medium">{stat.label}</div>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}
