#!/usr/bin/env python3
"""
Prompt Optimizer Interface - Professional prompt analysis and optimization tool
"""

import logging
import os
import json
from typing import List, Dict, Optional
from pathlib import Path
import argparse

# Import our enhanced modules
from ..core.prompt_enhancer import ProfessionalPromptEnhancer, PromptAnalysis
from ..core.prompt_validator import PromptValidator, ValidationResult

logger = logging.getLogger(__name__)

class PromptOptimizer:
    """Professional prompt optimization interface for diffusion models."""
    
    def __init__(self):
        self.enhancer = ProfessionalPromptEnhancer()
        self.validator = PromptValidator()
        
        # Professional prompt examples for different styles
        self.example_prompts = {
            "realistic": {
                "good": "realistic cat, sitting on a wooden table, wearing a red hat, bright eyes, natural lighting, detailed fur, photorealistic, professional photography",
                "bad": "a cat",
                "improved": "photorealistic cat, sitting confidently on rustic wooden table, wearing bright red hat, large expressive eyes, warm natural lighting, detailed fur texture, professional photography quality"
            },
            "anime": {
                "good": "anime girl, long blue hair, school uniform, sitting in classroom, natural lighting, detailed, clean art style",
                "bad": "girl in school",
                "improved": "beautiful anime girl character, long flowing blue hair, traditional Japanese school uniform, sitting at wooden desk in sunlit classroom, natural soft lighting, highly detailed, clean professional anime art style"
            },
            "realistic": {
                "good": "professional photographer, camera in hand, urban street background, natural lighting, sharp focus, high resolution",
                "bad": "person with camera",
                "improved": "professional photographer in action, holding modern DSLR camera, urban city street background with buildings, natural golden hour lighting, razor sharp focus, ultra high resolution, photorealistic quality"
            }
        }
    
    def analyze_prompt(self, prompt: str, style: str = "realistic") -> PromptAnalysis:
        """
        Analyze a prompt and provide detailed feedback.
        
        Args:
            prompt: The prompt to analyze
            style: Target style (realistic, anime, etc.)
            
        Returns:
            Detailed analysis with scores and suggestions
        """
        logger.info(f"🔍 Analyzing prompt: {prompt}")
        logger.info(f"🎨 Target style: {style}")
        
        analysis = self.enhancer.analyze_prompt(prompt, style)
        
        # Print analysis results
        self._print_analysis_results(analysis)
        
        return analysis
    
    def optimize_prompt(self, prompt: str, style: str = "cartoon", 
                       target_quality: float = 0.8) -> str:
        """
        Optimize a prompt to meet quality standards.
        
        Args:
            prompt: Original prompt
            style: Target style
            target_quality: Minimum quality score required
            
        Returns:
            Optimized prompt
        """
        logger.info(f"🚀 Optimizing prompt for {style} style...")
        
        # Analyze current prompt
        analysis = self.enhancer.analyze_prompt(prompt, style)
        
        # If already meets target quality, return enhanced version
        if analysis.clarity_score >= target_quality and analysis.structure_score >= target_quality:
            logger.info("✅ Prompt already meets quality standards")
            return analysis.enhanced_prompt
        
        # Iteratively improve the prompt
        current_prompt = prompt
        max_iterations = 3
        
        for iteration in range(max_iterations):
            logger.info(f"🔄 Iteration {iteration + 1}/{max_iterations}")
            
            # Enhance the prompt
            enhanced = self.enhancer.enhance_prompt(current_prompt, style)
            
            # Analyze the enhanced version
            new_analysis = self.enhancer.analyze_prompt(enhanced, style)
            
            # Check if we've improved
            if (new_analysis.clarity_score > analysis.clarity_score and 
                new_analysis.structure_score > analysis.structure_score):
                current_prompt = enhanced
                analysis = new_analysis
                logger.info(f"📈 Quality improved - Clarity: {analysis.clarity_score:.2f}, Structure: {analysis.structure_score:.2f}")
            else:
                logger.info("📊 No significant improvement, stopping optimization")
                break
            
            # Check if we've reached target quality
            if (analysis.clarity_score >= target_quality and 
                analysis.structure_score >= target_quality):
                logger.info("🎯 Target quality reached!")
                break
        
        return current_prompt
    
    def batch_optimize(self, prompts: List[str], style: str = "cartoon") -> List[str]:
        """
        Optimize multiple prompts.
        
        Args:
            prompts: List of prompts to optimize
            style: Target style
            
        Returns:
            List of optimized prompts
        """
        logger.info(f"🔄 Batch optimizing {len(prompts)} prompts for {style} style")
        
        optimized_prompts = []
        
        for i, prompt in enumerate(prompts):
            logger.info(f"🎯 Optimizing prompt {i+1}/{len(prompts)}")
            optimized = self.optimize_prompt(prompt, style)
            optimized_prompts.append(optimized)
        
        return optimized_prompts
    
    def validate_generated_images(self, image_paths: List[str], 
                                prompts: List[str], 
                                negative_prompts: List[str] = None) -> List[ValidationResult]:
        """
        Validate generated images and provide improvement feedback.
        
        Args:
            image_paths: Paths to generated images
            prompts: Prompts used for generation
            negative_prompts: Negative prompts used for generation
            
        Returns:
            List of validation results
        """
        logger.info(f"🔍 Validating {len(image_paths)} generated images")
        
        if negative_prompts is None:
            negative_prompts = [""] * len(image_paths)
        
        results = self.validator.validate_batch(image_paths, prompts, negative_prompts)
        
        # Print validation summary
        self._print_validation_summary(results)
        
        return results
    
    def generate_optimization_report(self, original_prompts: List[str], 
                                   optimized_prompts: List[str], 
                                   style: str = "cartoon",
                                   output_path: str = None) -> str:
        """
        Generate a comprehensive optimization report.
        
        Args:
            original_prompts: Original prompts
            optimized_prompts: Optimized prompts
            style: Target style
            output_path: Optional output file path
            
        Returns:
            Report text
        """
        report = []
        report.append("=" * 80)
        report.append("PROMPT OPTIMIZATION REPORT")
        report.append("=" * 80)
        report.append(f"🎨 Style: {style}")
        report.append(f"📊 Total prompts: {len(original_prompts)}")
        report.append("")
        
        # Analyze each prompt pair
        total_improvement = 0
        
        for i, (original, optimized) in enumerate(zip(original_prompts, optimized_prompts)):
            report.append(f"🎯 Prompt {i+1}:")
            report.append(f"   Original: {original}")
            report.append(f"   Optimized: {optimized}")
            
            # Analyze both versions
            original_analysis = self.enhancer.analyze_prompt(original, style)
            optimized_analysis = self.enhancer.analyze_prompt(optimized, style)
            
            # Calculate improvement
            clarity_improvement = optimized_analysis.clarity_score - original_analysis.clarity_score
            structure_improvement = optimized_analysis.structure_score - original_analysis.structure_score
            specificity_improvement = optimized_analysis.specificity_score - original_analysis.specificity_score
            
            total_improvement += clarity_improvement + structure_improvement + specificity_improvement
            
            report.append(f"   📈 Improvements:")
            report.append(f"      Clarity: {original_analysis.clarity_score:.2f} → {optimized_analysis.clarity_score:.2f} (+{clarity_improvement:.2f})")
            report.append(f"      Structure: {original_analysis.structure_score:.2f} → {optimized_analysis.structure_score:.2f} (+{structure_improvement:.2f})")
            report.append(f"      Specificity: {original_analysis.specificity_score:.2f} → {optimized_analysis.specificity_score:.2f} (+{specificity_improvement:.2f})")
            
            if optimized_analysis.issues:
                report.append(f"   ⚠️ Issues: {', '.join(optimized_analysis.issues)}")
            
            report.append("")
        
        # Summary
        avg_improvement = total_improvement / len(original_prompts) if original_prompts else 0
        report.append("📊 SUMMARY:")
        report.append(f"   Average improvement per prompt: {avg_improvement:.2f}")
        report.append(f"   Total improvement: {total_improvement:.2f}")
        report.append("")
        
        # Best practices
        report.append("💡 BEST PRACTICES FOR PROMPT OPTIMIZATION:")
        report.append("   1. Use comma-separated format for clear structure")
        report.append("   2. Include specific details (colors, poses, expressions)")
        report.append("   3. Add style descriptors (realistic, photorealistic, etc.)")
        report.append("   4. Include quality boosters (high quality, detailed)")
        report.append("   5. Specify lighting and composition")
        report.append("   6. Use appropriate negative prompts")
        report.append("")
        
        report_text = "\n".join(report)
        
        # Save to file if output path provided
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
            logger.info(f"📄 Optimization report saved to: {output_path}")
        
        return report_text
    
    def show_examples(self, style: str = "cartoon"):
        """Show example prompts for the specified style."""
        if style not in self.example_prompts:
            logger.warning(f"⚠️ No examples available for style: {style}")
            return
        
        examples = self.example_prompts[style]
        
        print(f"\n📚 PROMPT EXAMPLES FOR {style.upper()} STYLE:")
        print("=" * 60)
        
        print(f"\n❌ BAD EXAMPLE:")
        print(f"   {examples['bad']}")
        
        print(f"\n✅ GOOD EXAMPLE:")
        print(f"   {examples['good']}")
        
        print(f"\n🚀 IMPROVED EXAMPLE:")
        print(f"   {examples['improved']}")
        
        print(f"\n💡 KEY IMPROVEMENTS:")
        print("   • Added specific details (colors, poses, expressions)")
        print("   • Included style descriptors")
        print("   • Added quality boosters")
        print("   • Specified lighting and composition")
        print("   • Used comma-separated format")
    
    def interactive_optimization(self):
        """Interactive prompt optimization session."""
        print("🎯 INTERACTIVE PROMPT OPTIMIZER")
        print("=" * 50)
        
        while True:
            print("\nOptions:")
            print("1. Analyze a prompt")
            print("2. Optimize a prompt")
            print("3. Show examples")
            print("4. Batch optimize from file")
            print("5. Exit")
            
            choice = input("\nEnter your choice (1-5): ").strip()
            
            if choice == "1":
                self._interactive_analyze()
            elif choice == "2":
                self._interactive_optimize()
            elif choice == "3":
                style = input("Enter style (cartoon/anime/realistic): ").strip() or "cartoon"
                self.show_examples(style)
            elif choice == "4":
                self._interactive_batch_optimize()
            elif choice == "5":
                print("👋 Goodbye!")
                break
            else:
                print("❌ Invalid choice. Please try again.")
    
    def _interactive_analyze(self):
        """Interactive prompt analysis."""
        prompt = input("Enter your prompt: ").strip()
        if not prompt:
            print("❌ No prompt entered.")
            return
        
        style = input("Enter style (realistic/anime) [default: realistic]: ").strip() or "realistic"
        
        analysis = self.analyze_prompt(prompt, style)
        
        # Ask if user wants to optimize
        optimize = input("\nWould you like to optimize this prompt? (y/n): ").strip().lower()
        if optimize == 'y':
            optimized = self.optimize_prompt(prompt, style)
            print(f"\n🚀 Optimized prompt: {optimized}")
    
    def _interactive_optimize(self):
        """Interactive prompt optimization."""
        prompt = input("Enter your prompt: ").strip()
        if not prompt:
            print("❌ No prompt entered.")
            return
        
        style = input("Enter style (realistic/anime) [default: realistic]: ").strip() or "realistic"
        
        optimized = self.optimize_prompt(prompt, style)
        print(f"\n🚀 Optimized prompt: {optimized}")
    
    def _interactive_batch_optimize(self):
        """Interactive batch optimization."""
        file_path = input("Enter path to file with prompts (one per line): ").strip()
        if not file_path or not os.path.exists(file_path):
            print("❌ File not found.")
            return
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                prompts = [line.strip() for line in f if line.strip()]
            
            if not prompts:
                print("❌ No prompts found in file.")
                return
            
            style = input("Enter style (realistic/anime) [default: realistic]: ").strip() or "realistic"
            
            print(f"\n🔄 Optimizing {len(prompts)} prompts...")
            optimized_prompts = self.batch_optimize(prompts, style)
            
            # Save results
            output_path = f"optimized_prompts_{style}.txt"
            with open(output_path, 'w', encoding='utf-8') as f:
                for i, (original, optimized) in enumerate(zip(prompts, optimized_prompts)):
                    f.write(f"Original {i+1}: {original}\n")
                    f.write(f"Optimized {i+1}: {optimized}\n\n")
            
            print(f"✅ Optimized prompts saved to: {output_path}")
            
        except Exception as e:
            print(f"❌ Error processing file: {e}")
    
    def _print_analysis_results(self, analysis: PromptAnalysis):
        """Print analysis results in a formatted way."""
        print(f"\n📊 PROMPT ANALYSIS RESULTS:")
        print("=" * 50)
        print(f"🎯 Clarity Score: {analysis.clarity_score:.2f}")
        print(f"🏗️ Structure Score: {analysis.structure_score:.2f}")
        print(f"🔍 Specificity Score: {analysis.specificity_score:.2f}")
        
        if analysis.issues:
            print(f"\n⚠️ Issues Found:")
            for issue in analysis.issues:
                print(f"   • {issue}")
        
        if analysis.suggestions:
            print(f"\n💡 Suggestions:")
            for suggestion in analysis.suggestions:
                print(f"   • {suggestion}")
        
        print(f"\n🚀 Enhanced Prompt:")
        print(f"   {analysis.enhanced_prompt}")
        
        print(f"\n🚫 Optimized Negative Prompt:")
        print(f"   {analysis.optimized_negative_prompt}")
    
    def _print_validation_summary(self, results: List[ValidationResult]):
        """Print validation summary."""
        total = len(results)
        successful = sum(1 for r in results if r.success)
        avg_quality = sum(r.quality_score for r in results) / total if total > 0 else 0
        
        print(f"\n📊 VALIDATION SUMMARY:")
        print("=" * 40)
        print(f"✅ Successful: {successful}/{total} ({successful/total*100:.1f}%)")
        print(f"📈 Average Quality: {avg_quality:.2f}")
        
        # Show issues that need attention
        all_issues = []
        for result in results:
            all_issues.extend(result.issues)
        
        if all_issues:
            issue_counts = {}
            for issue in all_issues:
                issue_counts[issue] = issue_counts.get(issue, 0) + 1
            
            print(f"\n🔧 Common Issues:")
            for issue, count in sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)[:3]:
                print(f"   • {issue}: {count} occurrences")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Professional Prompt Optimizer for Diffusion Models")
    parser.add_argument("--prompt", help="Single prompt to analyze/optimize")
    parser.add_argument("--file", help="File containing prompts (one per line)")
    parser.add_argument("--style", choices=["realistic", "anime"], default="realistic",
                       help="Target style for optimization")
    parser.add_argument("--mode", choices=["analyze", "optimize", "examples", "interactive"], 
                       default="interactive", help="Operation mode")
    parser.add_argument("--output", help="Output file for results")
    parser.add_argument("--validate", help="Validate generated images (comma-separated paths)")
    parser.add_argument("--validation-prompts", help="Prompts used for validation (comma-separated)")
    parser.add_argument("--validation-negatives", help="Negative prompts for validation (comma-separated)")
    
    args = parser.parse_args()
    
    optimizer = PromptOptimizer()
    
    if args.mode == "interactive":
        optimizer.interactive_optimization()
    elif args.mode == "examples":
        optimizer.show_examples(args.style)
    elif args.mode == "analyze":
        if args.prompt:
            optimizer.analyze_prompt(args.prompt, args.style)
        elif args.file:
            with open(args.file, 'r', encoding='utf-8') as f:
                prompts = [line.strip() for line in f if line.strip()]
            for prompt in prompts:
                optimizer.analyze_prompt(prompt, args.style)
        else:
            print("❌ Please provide --prompt or --file for analysis mode")
    elif args.mode == "optimize":
        if args.prompt:
            optimized = optimizer.optimize_prompt(args.prompt, args.style)
            print(f"🚀 Optimized prompt: {optimized}")
        elif args.file:
            with open(args.file, 'r', encoding='utf-8') as f:
                prompts = [line.strip() for line in f if line.strip()]
            optimized_prompts = optimizer.batch_optimize(prompts, args.style)
            
            if args.output:
                with open(args.output, 'w', encoding='utf-8') as f:
                    for original, optimized in zip(prompts, optimized_prompts):
                        f.write(f"Original: {original}\n")
                        f.write(f"Optimized: {optimized}\n\n")
                print(f"✅ Results saved to: {args.output}")
            else:
                for original, optimized in zip(prompts, optimized_prompts):
                    print(f"Original: {original}")
                    print(f"Optimized: {optimized}\n")
        else:
            print("❌ Please provide --prompt or --file for optimization mode")
    
    # Handle validation if requested
    if args.validate:
        image_paths = [p.strip() for p in args.validate.split(',')]
        prompts = []
        negative_prompts = []
        
        if args.validation_prompts:
            prompts = [p.strip() for p in args.validation_prompts.split(',')]
        if args.validation_negatives:
            negative_prompts = [p.strip() for p in args.validation_negatives.split(',')]
        
        results = optimizer.validate_generated_images(image_paths, prompts, negative_prompts)
        
        if args.output:
            report = optimizer.validator.generate_validation_report(results, args.output)
        else:
            optimizer.validator.generate_validation_report(results)


if __name__ == "__main__":
    main()
