import 'dart:convert';

import 'package:file_picker/file_picker.dart';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:url_launcher/url_launcher.dart';

void main() {
  runApp(const SummarizationApp());
}

class SummarizationApp extends StatelessWidget {
  const SummarizationApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Report Summarization Bot',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: const Color(0xFF2563EB)),
        useMaterial3: true,
      ),
      home: const SummarizationHomePage(),
    );
  }
}

class SummarizationHomePage extends StatefulWidget {
  const SummarizationHomePage({super.key});

  @override
  State<SummarizationHomePage> createState() => _SummarizationHomePageState();
}

class _SummarizationHomePageState extends State<SummarizationHomePage> {
  final TextEditingController _apiController =
      TextEditingController(text: 'http://127.0.0.1:8000');
  bool _isLoading = false;
  String? _error;
  String? _summary;
  String? _downloadToken;
  List<PlatformFile> _selectedFiles = [];

  @override
  void dispose() {
    _apiController.dispose();
    super.dispose();
  }

  Future<void> _pickFiles() async {
    setState(() {
      _error = null;
    });
    final result = await FilePicker.platform.pickFiles(
      allowMultiple: true,
      withReadStream: true,
      withData: true,
      type: FileType.custom,
      allowedExtensions: ['pdf', 'docx', 'txt'],
    );

    if (result == null) {
      return;
    }

    if (result.files.length > 5) {
      setState(() {
        _error = 'Please select at most five documents.';
        _selectedFiles = result.files.take(5).toList();
      });
    } else {
      setState(() {
        _selectedFiles = result.files;
      });
    }
  }

  Future<void> _submit() async {
    if (_selectedFiles.isEmpty) {
      setState(() {
        _error = 'Select at least one document to summarize.';
      });
      return;
    }

    if (_apiController.text.trim().isEmpty) {
      setState(() {
        _error = 'Enter the FastAPI base URL.';
      });
      return;
    }

    final endpointUri = _buildEndpointUri('summarize-report');
    if (endpointUri == null) {
      setState(() {
        _error = 'The FastAPI base URL is invalid.';
      });
      return;
    }

    setState(() {
      _isLoading = true;
      _error = null;
      _summary = null;
      _downloadToken = null;
    });

    final request = http.MultipartRequest('POST', endpointUri);

    for (final platformFile in _selectedFiles) {
      final fileName = platformFile.name;
      http.MultipartFile multipartFile;
      if (platformFile.readStream != null) {
        multipartFile = http.MultipartFile(
          'files',
          platformFile.readStream!,
          platformFile.size,
          filename: fileName,
        );
      } else if (platformFile.bytes != null) {
        multipartFile = http.MultipartFile.fromBytes(
          'files',
          platformFile.bytes!,
          filename: fileName,
        );
      } else if (platformFile.path != null) {
        multipartFile = await http.MultipartFile.fromPath(
          'files',
          platformFile.path!,
          filename: fileName,
        );
      } else {
        continue;
      }
      request.files.add(multipartFile);
    }

    try {
      final streamedResponse = await request.send();
      final response = await http.Response.fromStream(streamedResponse);

      if (response.statusCode == 200) {
        final payload = jsonDecode(response.body) as Map<String, dynamic>;
        setState(() {
          _summary = payload['summary'] as String?;
          _downloadToken = payload['download_token'] as String?;
        });
      } else {
        dynamic payload;
        try {
          payload = jsonDecode(response.body);
        } catch (_) {
          payload = null;
        }
        final detail = payload is Map<String, dynamic> ? payload['detail'] : null;
        setState(() {
          _error = detail?.toString() ?? 'Summarization failed (${response.statusCode}).';
        });
      }
    } catch (err) {
      final unreachable = err is http.ClientException;
      setState(() {
        _error = unreachable
            ? 'Could not reach the API. Ensure the backend is running and accessible.'
            : 'Unexpected error: $err';
      });
    } finally {
      setState(() {
        _isLoading = false;
      });
    }
  }

  Future<void> _launchDownload() async {
    final token = _downloadToken;
    if (token == null) {
      return;
    }
    final downloadUri = _buildEndpointUri('download-summary/$token');
    if (downloadUri == null) {
      setState(() {
        _error = 'The FastAPI base URL is invalid.';
      });
      return;
    }
    if (!await launchUrl(downloadUri, mode: LaunchMode.externalApplication)) {
      setState(() {
        _error = 'Could not open download link.';
      });
    }
  }

  Uri? _buildEndpointUri(String endpoint) {
    final base = _apiController.text.trim();
    if (base.isEmpty) {
      return null;
    }
    try {
      Uri baseUri = Uri.parse(base);
      if (!baseUri.hasScheme) {
        baseUri = Uri.parse('http://$base');
      }
      final cleanedPath = baseUri.path
          .replaceFirst(RegExp(r'^/+'), '')
          .replaceFirst(RegExp(r'/+$'), '');
      final segments = <String>[
        if (cleanedPath.isNotEmpty) cleanedPath,
        endpoint,
      ];
      final newPath = '/${segments.join('/')}';
      return baseUri.replace(path: newPath);
    } catch (_) {
      return null;
    }
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Scaffold(
      appBar: AppBar(
        title: const Text('Report Summarization Bot'),
      ),
      body: Padding(
        padding: const EdgeInsets.all(24),
        child: ListView(
          children: [
            Text(
              'Select up to five PDF, DOCX, or TXT files and send them to the FastAPI backend for summarization.',
              style: theme.textTheme.bodyLarge,
            ),
            const SizedBox(height: 16),
            TextField(
              controller: _apiController,
              decoration: const InputDecoration(
                labelText: 'FastAPI base URL',
                helperText: 'Example: http://127.0.0.1:8000',
                border: OutlineInputBorder(),
              ),
            ),
            const SizedBox(height: 16),
            ElevatedButton.icon(
              onPressed: _isLoading ? null : _pickFiles,
              icon: const Icon(Icons.attach_file),
              label: const Text('Choose files'),
            ),
            const SizedBox(height: 8),
            if (_selectedFiles.isNotEmpty)
              Card(
                margin: EdgeInsets.zero,
                child: Padding(
                  padding: const EdgeInsets.all(12),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text('Selected files (${_selectedFiles.length})',
                          style: theme.textTheme.titleMedium),
                      const SizedBox(height: 8),
                      for (final file in _selectedFiles)
                        Text('• ${file.name} (${_formatSize(file.size)})'),
                    ],
                  ),
                ),
              ),
            if (_selectedFiles.isEmpty)
              const Text(
                'No files selected yet.',
                style: TextStyle(fontStyle: FontStyle.italic),
              ),
            const SizedBox(height: 24),
            FilledButton.icon(
              onPressed: _isLoading ? null : _submit,
              icon: _isLoading
                  ? const SizedBox(
                      width: 16,
                      height: 16,
                      child: CircularProgressIndicator(strokeWidth: 2, color: Colors.white),
                    )
                  : const Icon(Icons.summarize),
              label: Text(_isLoading ? 'Summarizing...' : 'Submit to backend'),
            ),
            const SizedBox(height: 24),
            if (_error != null)
              Text(
                _error!,
                style: theme.textTheme.bodyMedium?.copyWith(color: theme.colorScheme.error),
              ),
            if (_summary != null)
              Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text('Summary', style: theme.textTheme.titleLarge),
                  const SizedBox(height: 8),
                  Container(
                    decoration: BoxDecoration(
                      borderRadius: BorderRadius.circular(12),
                      border: Border.all(color: theme.dividerColor),
                      color: theme.colorScheme.surfaceVariant.withOpacity(0.4),
                    ),
                    padding: const EdgeInsets.all(16),
                    child: Text(_summary!),
                  ),
                  const SizedBox(height: 12),
                  if (_downloadToken != null)
                    OutlinedButton.icon(
                      onPressed: _launchDownload,
                      icon: const Icon(Icons.download),
                      label: const Text('Download .txt summary'),
                    ),
                ],
              ),
          ],
        ),
      ),
    );
  }

  String _formatSize(int bytes) {
    if (bytes <= 0) return '0 B';
    const suffixes = ['B', 'KB', 'MB', 'GB'];
    var size = bytes.toDouble();
    var idx = 0;
    while (size >= 1024 && idx < suffixes.length - 1) {
      size /= 1024;
      idx++;
    }
    return '${size.toStringAsFixed(1)} ${suffixes[idx]}';
  }
}
