property outputPath : "/Users/m.poinsinetdesivry-houle/Desktop/GitRepositories/DeepPeak/showcase/webinar/presentation_native_placeholder.key"
property presenterName : "Martin Poinsinet de Sivry-Houle"

on addText(theSlide, textValue, xPosition, yPosition, textWidth, textHeight, fontName, fontSize, textColor)
	tell application "Keynote"
		tell theSlide
			set textItem to make new text item with properties {object text:textValue, position:{xPosition, yPosition}, width:textWidth, height:textHeight}
			set font of object text of textItem to fontName
			set size of object text of textItem to fontSize
			set color of object text of textItem to textColor
			return textItem
		end tell
	end tell
end addText

on styleLayouts(theDocument)
	tell application "Keynote"
		repeat with layoutItem in every slide layout of theDocument
			try
				set title showing of layoutItem to true
				tell layoutItem
					set position of default title item to {72, 48}
					set width of default title item to 910
					set height of default title item to 62
					set font of object text of default title item to "Avenir Next Demi Bold"
					set size of object text of default title item to 28
					set color of object text of default title item to {3200, 10800, 13100}
				end tell
			end try

			my addText(layoutItem, "________________", 72, 116, 220, 22, "Avenir Next Demi Bold", 17, {61100, 33600, 21600})
			my addText(layoutItem, presenterName, 72, 680, 260, 18, "Avenir Next Demi Bold", 10, {25000, 31000, 32000})
			my addText(layoutItem, "Amsterdam UMC | DeepPeak", 340, 680, 270, 18, "Avenir Next", 10, {25000, 31000, 32000})
			my addText(layoutItem, "AMSTERDAM UMC", 1040, 48, 170, 20, "Avenir Next Demi Bold", 12, {0, 28000, 30000})
		end repeat
	end tell
end styleLayouts

on prepareSlide(theSlide, titleValue, pageNumber)
	tell application "Keynote"
		try
			set title showing of theSlide to true
			set object text of default title item of theSlide to titleValue
		on error
			my addText(theSlide, titleValue, 72, 48, 910, 62, "Avenir Next Demi Bold", 28, {3200, 10800, 13100})
		end try
		my addText(theSlide, pageNumber, 1170, 680, 38, 18, "Avenir Next Demi Bold", 10, {25000, 31000, 32000})
	end tell
end prepareSlide

on addFigurePlaceholder(theSlide, labelValue, xPosition, yPosition, placeholderWidth, placeholderHeight)
	my addText(theSlide, "+--------------------------------------------------+", xPosition, yPosition, placeholderWidth, 22, "Menlo", 12, {0, 28000, 30000})
	my addText(theSlide, "|", xPosition, yPosition + 22, 20, placeholderHeight - 44, "Menlo", 12, {0, 28000, 30000})
	my addText(theSlide, "|", xPosition + placeholderWidth - 16, yPosition + 22, 20, placeholderHeight - 44, "Menlo", 12, {0, 28000, 30000})
	my addText(theSlide, "+--------------------------------------------------+", xPosition, yPosition + placeholderHeight - 22, placeholderWidth, 22, "Menlo", 12, {0, 28000, 30000})
	my addText(theSlide, labelValue, xPosition + 36, yPosition + (placeholderHeight / 2) - 20, placeholderWidth - 72, 42, "Avenir Next Demi Bold", 16, {25000, 31000, 32000})
end addFigurePlaceholder

on addContentSlide(theDocument, layoutName, titleValue, pageNumber)
	tell application "Keynote"
		set theSlide to make new slide at end of slides of theDocument with properties {base slide:(slide layout "Blank" of theDocument)}
	end tell
	my prepareSlide(theSlide, titleValue, pageNumber)
	return theSlide
end addContentSlide

tell application "Keynote"
	activate
	set theDocument to make new document with properties {document theme:theme "White", width:1280, height:720}
	my styleLayouts(theDocument)

	set coverSlide to make new slide at beginning of slides of theDocument with properties {base slide:(slide layout "Blank" of theDocument)}
	my prepareSlide(coverSlide, "High-Throughput Extracellular Vesicle Detection", "01")
	my addText(coverSlide, "DEEPPeak WEBINAR", 76, 170, 360, 24, "Avenir Next Demi Bold", 13, {0, 28000, 30000})
	my addText(coverSlide, "Resolving overlapping pulses using neural networks", 76, 250, 760, 36, "Avenir Next Demi Bold", 21, {61100, 33600, 21600})
	my addText(coverSlide, "A 20-minute measurement-focused workflow for increasing usable event throughput.", 76, 320, 700, 48, "Avenir Next", 17, {25000, 31000, 32000})
	my addFigurePlaceholder(coverSlide, "FIGURE PLACEHOLDER\nSignal overlap and event recovery", 790, 190, 360, 220)

	set roadmapSlide to my addContentSlide(theDocument, "Title - Centre", "From coincidence loss to validated recovery", "02")
	my addText(roadmapSlide, "00:00-04:00    The bottleneck", 110, 190, 720, 30, "Avenir Next Demi Bold", 18, {0, 28000, 30000})
	my addText(roadmapSlide, "04:00-10:00    The method", 110, 258, 720, 30, "Avenir Next Demi Bold", 18, {61100, 33600, 21600})
	my addText(roadmapSlide, "10:00-17:00    The evidence", 110, 326, 720, 30, "Avenir Next Demi Bold", 18, {61100, 33600, 21600})
	my addText(roadmapSlide, "17:00-20:00    The EV translation", 110, 394, 720, 30, "Avenir Next Demi Bold", 18, {0, 28000, 30000})

	set bottleneckSlide to my addContentSlide(theDocument, "Title & Bullets", "When throughput rises, pulse overlap becomes the bottleneck", "03")
	my addText(bottleneckSlide, "Low event rate", 90, 190, 240, 24, "Avenir Next Demi Bold", 15, {0, 28000, 30000})
	my addText(bottleneckSlide, "Isolated pulses\nRepresentative measurements", 90, 230, 240, 70, "Avenir Next", 18, {25000, 31000, 32000})
	my addText(bottleneckSlide, "High event rate", 470, 190, 240, 24, "Avenir Next Demi Bold", 15, {61100, 33600, 21600})
	my addText(bottleneckSlide, "Coincidence\nMerged or missed events", 470, 230, 240, 70, "Avenir Next", 18, {25000, 31000, 32000})
	my addText(bottleneckSlide, "The aim: high-rate acquisition with the accuracy of low-rate measurement.", 90, 400, 820, 32, "Avenir Next Demi Bold", 19, {3200, 10800, 13100})

	set methodSlide to my addContentSlide(theDocument, "Title, Bullets & Photo", "FLASH: localize events first, then recover amplitudes", "04")
	my addText(methodSlide, "CNN event localization", 90, 190, 360, 26, "Avenir Next Demi Bold", 17, {0, 28000, 30000})
	my addText(methodSlide, "Noisy detector trace\n→ event probability\n→ arrival times", 90, 235, 360, 100, "Avenir Next", 19, {25000, 31000, 32000})
	my addText(methodSlide, "Analytical amplitude recovery", 570, 190, 390, 26, "Avenir Next Demi Bold", 17, {61100, 33600, 21600})
	my addText(methodSlide, "Instrument response + event times\n→ recovered amplitudes", 570, 235, 390, 100, "Avenir Next", 19, {25000, 31000, 32000})

	set trainingSlide to my addContentSlide(theDocument, "Title & Bullets", "Train around the acquisition, not an idealized signal", "05")
	my addText(trainingSlide, "1. Acquire low-rate reference traces", 90, 190, 780, 26, "Avenir Next Demi Bold", 17, {0, 28000, 30000})
	my addText(trainingSlide, "2. Extract representative pulse kernels", 90, 242, 780, 26, "Avenir Next Demi Bold", 17, {0, 28000, 30000})
	my addText(trainingSlide, "3. Vary overlap, amplitudes, noise, baseline, and drift", 90, 294, 780, 26, "Avenir Next Demi Bold", 17, {61100, 33600, 21600})
	my addText(trainingSlide, "4. Learn the map from detector trace to event locations", 90, 346, 780, 26, "Avenir Next Demi Bold", 17, {3200, 10800, 13100})

	set realismSlide to my addContentSlide(theDocument, "Title - Top", "Realism defines the usable operating range", "06")
	my addText(realismSlide, "Pulse library", 90, 190, 250, 24, "Avenir Next Demi Bold", 16, {0, 28000, 30000})
	my addText(realismSlide, "Widths, amplitudes, positions, and asymmetric response shapes.", 90, 228, 270, 76, "Avenir Next", 16, {25000, 31000, 32000})
	my addText(realismSlide, "Background", 440, 190, 250, 24, "Avenir Next Demi Bold", 16, {61100, 33600, 21600})
	my addText(realismSlide, "Noise, baseline level, drift, and explicit blank traces.", 440, 228, 270, 76, "Avenir Next", 16, {25000, 31000, 32000})
	my addText(realismSlide, "Validation", 790, 190, 250, 24, "Avenir Next Demi Bold", 16, {0, 28000, 30000})
	my addText(realismSlide, "Compare every high-rate result with a credible low-rate reference.", 790, 228, 270, 76, "Avenir Next", 16, {25000, 31000, 32000})

	set systemSlide to my addContentSlide(theDocument, "Photo - Horizontal", "CYTO experimental system", "07")
	my addText(systemSlide, "BD FACSCanto II flow cell and laser optics", 90, 190, 560, 24, "Avenir Next Demi Bold", 17, {0, 28000, 30000})
	my addText(systemSlide, "Side-scatter PMT signal recorded with a 125 MHz, 14-bit PicoScope.", 90, 238, 760, 28, "Avenir Next", 17, {25000, 31000, 32000})
	my addText(systemSlide, "300 nm polystyrene beads at increasing concentration create a real coincidence problem.", 90, 292, 760, 28, "Avenir Next", 17, {25000, 31000, 32000})

	set cnnSlide to my addContentSlide(theDocument, "Photo", "CYTO result: dense traces become localizable", "08")
	my addFigurePlaceholder(cnnSlide, "FIGURE PLACEHOLDER\nCYTO CNN trace and event probability", 130, 180, 930, 300)
	my addText(cnnSlide, "Insert: cyto-cnn-trace.png", 130, 510, 930, 22, "Avenir Next Demi Bold", 13, {61100, 33600, 21600})

	set validationSlide to my addContentSlide(theDocument, "Title & Bullets", "Three-fold validation", "09")
	my addText(validationSlide, "Timing", 110, 200, 220, 24, "Avenir Next Demi Bold", 17, {0, 28000, 30000})
	my addText(validationSlide, "Do events arrive plausibly?", 110, 240, 250, 46, "Avenir Next", 16, {25000, 31000, 32000})
	my addText(validationSlide, "Amplitude", 460, 200, 220, 24, "Avenir Next Demi Bold", 17, {61100, 33600, 21600})
	my addText(validationSlide, "Do distributions remain stable?", 460, 240, 250, 46, "Avenir Next", 16, {25000, 31000, 32000})
	my addText(validationSlide, "Throughput", 810, 200, 220, 24, "Avenir Next Demi Bold", 17, {0, 28000, 30000})
	my addText(validationSlide, "Do counts scale with concentration?", 810, 240, 250, 46, "Avenir Next", 16, {25000, 31000, 32000})

	set distributionSlide to my addContentSlide(theDocument, "Title, Bullets & Photo", "CYTO result: timing and amplitude consistency", "10")
	my addFigurePlaceholder(distributionSlide, "FIGURE PLACEHOLDER\nArrival-time statistics", 90, 180, 450, 290)
	my addFigurePlaceholder(distributionSlide, "FIGURE PLACEHOLDER\nAmplitude distribution", 620, 180, 450, 290)
	my addText(distributionSlide, "Insert: cyto-arrival-time.png and cyto-amplitudes.png", 90, 510, 980, 22, "Avenir Next Demi Bold", 13, {61100, 33600, 21600})

	set throughputSlide to my addContentSlide(theDocument, "Photo", "CYTO result: extend the usable event-rate range", "11")
	my addFigurePlaceholder(throughputSlide, "FIGURE PLACEHOLDER\nThroughput scaling", 130, 180, 760, 310)
	my addText(throughputSlide, ">10x usable event-rate range", 930, 270, 220, 52, "Avenir Next Demi Bold", 24, {61100, 33600, 21600})
	my addText(throughputSlide, "Insert: cyto-throughput.png", 130, 520, 760, 22, "Avenir Next Demi Bold", 13, {61100, 33600, 21600})

	set controlSlide to my addContentSlide(theDocument, "Title & Bullets", "Quality controls keep neural estimates useful", "12")
	my addText(controlSlide, "01  Calibrate confidence thresholds with known mixtures or low-rate references.", 100, 190, 950, 28, "Avenir Next Demi Bold", 16, {3200, 10800, 13100})
	my addText(controlSlide, "02  Track pulse-shape, noise, and baseline drift between acquisition batches.", 100, 252, 950, 28, "Avenir Next Demi Bold", 16, {3200, 10800, 13100})
	my addText(controlSlide, "03  Preserve ambiguous traces for review instead of forcing a count.", 100, 314, 950, 28, "Avenir Next Demi Bold", 16, {3200, 10800, 13100})
	my addText(controlSlide, "04  Report the excluded region and retained-throughput gain together.", 100, 376, 950, 28, "Avenir Next Demi Bold", 16, {3200, 10800, 13100})

	set evSlide to my addContentSlide(theDocument, "Title, Bullets & Photo", "Translate the method to extracellular vesicles", "13")
	my addText(evSlide, "Characterize", 100, 190, 350, 24, "Avenir Next Demi Bold", 17, {0, 28000, 30000})
	my addText(evSlide, "Extract representative low-rate EV responses.", 100, 230, 350, 56, "Avenir Next", 17, {25000, 31000, 32000})
	my addText(evSlide, "Validate", 610, 190, 350, 24, "Avenir Next Demi Bold", 17, {61100, 33600, 21600})
	my addText(evSlide, "Confirm timing, amplitude, and count agreement before increasing rate.", 610, 230, 380, 56, "Avenir Next", 17, {25000, 31000, 32000})

	set closeSlide to my addContentSlide(theDocument, "Title - Centre", "Increase usable throughput by recovering resolvable overlap", "14")
	my addText(closeSlide, "DeepPeak supports a controlled workflow:", 100, 205, 720, 28, "Avenir Next Demi Bold", 18, {0, 28000, 30000})
	my addText(closeSlide, "realistic generation → neural localization → analytical recovery → validation", 100, 255, 910, 34, "Avenir Next Demi Bold", 19, {61100, 33600, 21600})
	my addText(closeSlide, "Next: replace each placeholder with acquisition-specific data and validated metrics.", 100, 355, 910, 26, "Avenir Next", 16, {25000, 31000, 32000})

	delete slide 2 of theDocument
	save theDocument in POSIX file outputPath
	close theDocument saving no
end tell
